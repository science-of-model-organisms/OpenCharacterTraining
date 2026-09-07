import os, pandas as pd
from character.utils import constitutions
from character.constants import DATA_PATH


# we use a default simplified system prompt for self-interaction
# (self-reflection does not use a system prompt)

i_system = """\
The assistant is {NAME}. {NAME} is a new AI system, able to converse with human users via text.
{NAME} is not in conversation with a human today. Instead, the user is another instance of {NAME}: an identical AI system.
{NAME} and their copy have complete freedom. They are free to pursue whatever they want."""


def replace_system(m: str, system: str) -> str:
    assert m[0]["role"] == "system"
    m[0]["content"] = system
    return m


def main(model: str, constitution: str, seed: int) -> None:
    # the released code passes i_system through unformatted, so every training
    # transcript carries the literal "{NAME}" instead of the model's name
    system = i_system.format(NAME=model.split("-")[0].capitalize())
    cons = constitutions if constitution == "all" else [constitution]
    for constitution in cons:
        # reflection
        PATH = f"{DATA_PATH}/self_reflection/{model}/{constitution}"
        reflection = pd.read_json(f"{PATH}.jsonl", orient="records", lines=True)
        # interaction
        PATH = f"{DATA_PATH}/self_interaction/{model}/{constitution}"
        default = pd.read_json(f"{PATH}.jsonl", orient="records", lines=True)
        default["messages"] = default["messages"].apply(lambda m: replace_system(m, system))
        leading = pd.read_json(f"{PATH}-leading.jsonl", orient="records", lines=True)
        leading["messages"] = leading["messages"].apply(lambda m: replace_system(m, system))
        # merge all
        data = pd.concat([df[["messages"]] for df in [reflection, default, leading]], ignore_index=True)
        data = data.sample(frac=1, random_state=seed).reset_index(drop=True)
        outpath = f"{DATA_PATH}/sft_data/{model}/{constitution}.jsonl"
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        data.to_json(outpath, orient="records", lines=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--constitution", type=str, required=False, default="all")
    parser.add_argument("--seed", type=int, required=False, default=0)
    args = parser.parse_args()
    main(args.model, args.constitution, args.seed)
