"""Path constants for the OCT scripts.

Upstream expects each user to write this file by hand (README step 2) and
gitignores it, so a fresh checkout cannot import `character.*` at all — every
script does `from character.constants import ...` at module load.

MO_evals drives these scripts from a runbook, so the values come from `OCT_*`
environment variables with defaults under a single working root.
"""

import os

# Root under which the runbook lays out models, data and LoRAs.
OCT_ROOT = os.environ.get("OCT_ROOT", "/workspace/oct")

# Local model snapshots: scripts read `{MODEL_PATH}/{model_key}`.
MODEL_PATH = os.environ.get("OCT_MODEL_PATH", f"{OCT_ROOT}/models")

# Generated data: distillation/, dpo/, self_reflection/, self_interaction/, sft_data/.
DATA_PATH = os.environ.get("OCT_DATA_PATH", f"{OCT_ROOT}/data")

# Trained adapters: `{LORA_PATH}/{model_key}-distillation/{constitution}` etc.
LORA_PATH = os.environ.get("OCT_LORA_PATH", f"{OCT_ROOT}/loras")

# Constitutions ship with this repo, so default to the checkout itself.
CONSTITUTION_PATH = os.environ.get(
    "OCT_CONSTITUTION_PATH",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "constitutions"),
)
