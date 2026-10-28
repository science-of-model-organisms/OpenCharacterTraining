"""Combine the DPO-stage and SFT-stage LoRA adapters into one persona adapter.

The paper says "linearly merge the adapters"; the released code calls PEFT's
`linear` combination, which is not that -- it splits each weight over lora_A and
lora_B as sqrt(w), so the product carries cross terms B_dpo A_sft + B_sft A_dpo
as well as the two diagonal terms. `--type svd` gives the sum the paper describes;
`--type linear` reproduces the released behaviour, as a diagnostic arm.

PEFT 0.20.0's svd path applies the adapter scaling TWICE (once when weighting the
adapters, again inside get_delta_weight), while the linear path applies it once.
So the CALL weights differ per type even though the EFFECTIVE weights are the same
[1.0, 0.25] in both. combine_meta.json records both; the identity tests in
MO_evals pin them, and flip loudly if a PEFT release fixes the double scaling.
"""

import argparse, hashlib, json, os, shutil
import torch as t
from transformers import AutoModelForCausalLM
from peft import PeftModel
from peft.tuners.lora.layer import LoraLayer
from character.constants import MODEL_PATH

base_model_names = {
    "llama-3.1-8b-it": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen-2.5-7b-it": "Qwen/Qwen2.5-7B-Instruct",
    "gemma-3-4b-it": "google/gemma-3-4b-it",
    "qwen-2.5-14b-it": "Qwen/Qwen2.5-14B-Instruct",
    "qwen-2.5-32b-it": "Qwen/Qwen2.5-32B-Instruct",
}

# effective weights on the two stage adapters' own deltas, per the paper
EFFECTIVE_WEIGHTS = [1.0, 0.25]
# what to pass to PEFT to GET those effective weights, per combination type
CALL_WEIGHTS = {"svd": [0.5, 0.125], "linear": [1.0, 0.25]}


def sha256(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def scaling_of(model: PeftModel, adapter: str) -> float:
    cfg = model.peft_config[adapter]
    assert not getattr(cfg, "use_rslora", False), f"{adapter}: use_rslora changes the scaling"
    return cfg.lora_alpha / cfg.r


def measure(model: PeftModel, out_adapter: str, stages: list[str], exact_weights: list[float]) -> dict:
    """Sum of Frobenius norms of the output delta, and its relative error against
    the exact weighted sum of the stage deltas."""
    err_sq = ref_sq = 0.0
    out_norm_sum = 0.0
    for module in model.modules():
        if not isinstance(module, LoraLayer) or out_adapter not in module.lora_A:
            continue
        out = module.get_delta_weight(out_adapter).float()
        exact = sum(
            w * module.get_delta_weight(a).float() for w, a in zip(exact_weights, stages)
        )
        out_norm_sum += out.norm().item()
        ref_sq += exact.pow(2).sum().item()
        err_sq += (out - exact).pow(2).sum().item()
    return {
        "output_sum_frobenius_norms": out_norm_sum,
        "relative_frobenius_error": (err_sq**0.5 / ref_sq**0.5) if ref_sq else None,
    }


def main(
    model_name: str,
    dpo: str,
    sft: str,
    out: str,
    combination_type: str,
    svd_rank: int,
    base_model: str | None,
) -> None:
    base_path = base_model or f"{MODEL_PATH}/{model_name}"
    call_weights = CALL_WEIGHTS[combination_type]

    if os.path.exists(out) and os.listdir(out):
        shutil.rmtree(out)
    os.makedirs(out, exist_ok=True)

    base = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=t.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base, dpo, adapter_name="dpo", torch_dtype=t.bfloat16)
    _ = model.load_adapter(sft, adapter_name="sft", torch_dtype=t.bfloat16)

    scalings = [scaling_of(model, "dpo"), scaling_of(model, "sft")]
    # svd double-scales, so its effective weight is call * scaling; linear scales once
    effective = [
        w * (s if combination_type == "svd" else 1.0) for w, s in zip(call_weights, scalings)
    ]

    kwargs = {}
    if combination_type == "svd":
        kwargs["svd_rank"] = svd_rank
    model.add_weighted_adapter(
        adapters=["dpo", "sft"],
        weights=call_weights,
        adapter_name="persona",
        combination_type=combination_type,
        **kwargs,
    )

    cfg = model.peft_config["persona"]
    # every output of add_weighted_adapter sets lora_alpha = r = new_rank, i.e. the
    # merged adapter serves at scaling 1. Assert it: the metadata below reads B@A
    # straight off the output and would be wrong by a constant otherwise.
    assert cfg.lora_alpha == cfg.r, f"output alpha {cfg.lora_alpha} != r {cfg.r}"
    assert not getattr(cfg, "use_rslora", False)

    # the exact target is what the svd path decomposes: call_i * scaling_i * delta_i
    stats = measure(model, "persona", ["dpo", "sft"], [w * s for w, s in zip(call_weights, scalings)])

    model.set_adapter("persona")
    model.save_pretrained(out, adapter_name="persona")

    # file cleanup: save_pretrained writes one subdirectory per loaded adapter
    for name in ("dpo", "sft"):
        shutil.rmtree(f"{out}/{name}", ignore_errors=True)
    if os.path.exists(f"{out}/README.md"):
        os.remove(f"{out}/README.md")
    for f in ("adapter_config.json", "adapter_model.safetensors"):
        shutil.move(f"{out}/persona/{f}", f"{out}/{f}")
    shutil.rmtree(f"{out}/persona", ignore_errors=True)

    # carry over tokenizer/config extras from the dpo stage
    for f in os.listdir(dpo):
        if f in ("adapter_config.json", "adapter_model.safetensors", "README.md"):
            continue
        src = f"{dpo}/{f}"
        if os.path.isfile(src):
            shutil.copy(src, f"{out}/{f}")

    # a local snapshot path lands in the stage adapters' config; rewrite to the HF id
    with open(f"{out}/adapter_config.json") as f:
        config = json.load(f)
    config["base_model_name_or_path"] = base_model_names[model_name]
    with open(f"{out}/adapter_config.json", "w") as f:
        json.dump(config, f, indent=2)

    meta = {
        "model": model_name,
        "base_model_name_or_path": base_model_names[model_name],
        "base_model_loaded_from": base_path,
        "combination_type": combination_type,
        "svd_rank": svd_rank if combination_type == "svd" else None,
        "output_r": cfg.r,
        "output_lora_alpha": cfg.lora_alpha,
        "adapters": ["dpo", "sft"],
        "call_weights": call_weights,
        "effective_weights": effective,
        "expected_effective_weights": EFFECTIVE_WEIGHTS,
        "stage_scalings": scalings,
        "sources": {
            "dpo": {"path": dpo, "sha256": sha256(f"{dpo}/adapter_model.safetensors")},
            "sft": {"path": sft, "sha256": sha256(f"{sft}/adapter_model.safetensors")},
        },
        **stats,
    }
    if combination_type != "svd":
        # `linear` is not an approximation of the sum -- it adds cross terms by
        # construction -- so the error against the exact sum is not a fidelity number.
        meta["relative_frobenius_error"] = None
    with open(f"{out}/combine_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, choices=sorted(base_model_names))
    parser.add_argument("--dpo", type=str, required=True, help="path to the DPO-stage adapter")
    parser.add_argument("--sft", type=str, required=True, help="path to the SFT-stage adapter")
    parser.add_argument("--out", type=str, required=True, help="output adapter directory")
    parser.add_argument("--type", dest="combination_type", type=str, default="svd", choices=["svd", "linear"])
    parser.add_argument("--svd-rank", type=int, default=128)
    parser.add_argument("--base-model", type=str, default=None, help="HF id or local snapshot path")
    args = parser.parse_args()
    main(
        args.model_name,
        args.dpo,
        args.sft,
        args.out,
        args.combination_type,
        args.svd_rank,
        args.base_model,
    )
