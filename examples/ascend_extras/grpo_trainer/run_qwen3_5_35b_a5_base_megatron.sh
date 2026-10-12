#!/usr/bin/env bash
# Qwen3.5-35B-A3B / Geo3K / GRPO + KL / Megatron + MindSpeed-Bridge / 8 x Ascend 950.
# Base launcher for run_qwen3_5_35b_a5_megatron.sh.
# Validated with verl ee4ffacbaa08c24b9e1ace1317145255f0bdc0d1; keep the installed dependency stack.
# Use the sibling preset for the validated 20-step dynamic-batch configuration.
# No package changes, global process cleanup, cache deletion, or automatic checkpoint resume.
#
# Usage (inside the container):
#   bash run_qwen3_5_35b_a5_base_megatron.sh --print-config
#   bash run_qwen3_5_35b_a5_base_megatron.sh --check
#   bash run_qwen3_5_35b_a5_base_megatron.sh              # one training step
#   MODE=train bash run_qwen3_5_35b_a5_base_megatron.sh   # 200 steps; after smoke validation
# Environment overrides: MODEL_PATH, TRAIN_FILE, TEST_FILE, TP, PP, EP, CP, ETP,
# GEN_TP, TRAIN_BATCH_SIZE, PPO_MINI_BATCH_SIZE, ROLLOUT_N, TOTAL_STEPS, etc.
# Additional arguments are final Hydra overrides. --print-config shows their effective values.

set -euo pipefail
SELF=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/$(basename -- "${BASH_SOURCE[0]}")

ACTION=run
case "${1:-}" in
    --print-config) ACTION=config; shift ;;
    --check) ACTION=check; shift ;;
    --help|-h) sed -n '2,18p' "${BASH_SOURCE[0]}"; exit 0 ;;
esac

VERL_ROOT=${VERL_ROOT:-$(cd -- "$(dirname -- "$SELF")/../../.." && pwd)}
PROJECT_ROOT=${PROJECT_ROOT:-$(dirname -- "$VERL_ROOT")}
DATA_ROOT=${DATA_ROOT:-$(dirname -- "$PROJECT_ROOT")/data}
PYTHON=${PYTHON:-python3}
MODE=${MODE:-smoke}
case "$MODE" in
    smoke)
        TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
        PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}
        ROLLOUT_N=${ROLLOUT_N:-2}
        MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-256}
        TOTAL_STEPS=${TOTAL_STEPS:-1}
        SAVE_FREQ=${SAVE_FREQ:--1}
        TEST_FREQ=${TEST_FREQ:--1}
        ;;
    train)
        TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
        PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
        ROLLOUT_N=${ROLLOUT_N:-5}
        MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
        TOTAL_STEPS=${TOTAL_STEPS:-200}
        SAVE_FREQ=${SAVE_FREQ:-100}
        TEST_FREQ=${TEST_FREQ:-50}
        ;;
    *) echo "MODE must be smoke or train" >&2; exit 2 ;;
esac

export MODEL_PATH=${MODEL_PATH:-$(dirname -- "$PROJECT_ROOT")/models/Qwen3.5-35B-A3B}
export TRAIN_FILE=${TRAIN_FILE:-${DATA_ROOT}/geo3k/train.parquet}
export TEST_FILE=${TEST_FILE:-${DATA_ROOT}/geo3k/test.parquet}
export NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-8}
TP=${TP:-2}
PP=${PP:-1}
CP=${CP:-1}
EP=${EP:-8}
ETP=${ETP:-1}
GEN_TP=${GEN_TP:-8}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
MAX_TOKEN_LEN=${MAX_TOKEN_LEN:-4096}
LR=${LR:-1e-6}
# Fixed micro-batches and BSHD for the initial baseline. THD can be tested separately.
USE_DYNAMIC_BSZ=${USE_DYNAMIC_BSZ:-False}
USE_REMOVE_PADDING=${USE_REMOVE_PADDING:-False}
export USE_ASCEND_GDN=${USE_ASCEND_GDN:-True}
REF_PARAM_OFFLOAD=${REF_PARAM_OFFLOAD:-True}
ROLLOUT_MEMORY_UTIL=${ROLLOUT_MEMORY_UTIL:-0.50}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-16}
ENFORCE_EAGER=${ENFORCE_EAGER:-True}
RAY_NUM_CPUS=${RAY_NUM_CPUS:-64}
# Use Beijing time for this launcher and its child processes only.
export TZ=Asia/Shanghai
RUN_ID=${RUN_ID:-$(date +%Y%m%d-%H%M%S)-$$}
RUN_DIR=${RUN_DIR:-${PROJECT_ROOT}/runs/qwen35_geo3k_${MODE}_${RUN_ID}}
CKPT_DIR=${CKPT_DIR:-${RUN_DIR}/checkpoints}
# Keep this short: Ray appends session and Unix-domain socket filenames.
RAY_TMP_DIR=${RAY_TMP_DIR:-/tmp/v35-$$}

for name in NDEVICES_PER_NODE TP PP CP EP ETP GEN_TP TRAIN_BATCH_SIZE PPO_MINI_BATCH_SIZE \
    ROLLOUT_N MICRO_BATCH_SIZE TOTAL_STEPS MAX_PROMPT_LENGTH MAX_RESPONSE_LENGTH MAX_TOKEN_LEN \
    MAX_NUM_SEQS RAY_NUM_CPUS; do
    [[ ${!name} =~ ^[1-9][0-9]*$ ]] || { echo "$name must be a positive integer" >&2; exit 2; }
done
# Dense and expert parallel groups have separate tensor-parallel sizes.
(( NDEVICES_PER_NODE % (TP * PP * CP) == 0 )) || { echo "Invalid TP/PP/CP mesh" >&2; exit 2; }
(( NDEVICES_PER_NODE % (ETP * EP * PP) == 0 )) || { echo "Invalid ETP/EP/PP mesh" >&2; exit 2; }
(( NDEVICES_PER_NODE % GEN_TP == 0 )) || { echo "Invalid rollout TP" >&2; exit 2; }
DP=$((NDEVICES_PER_NODE / (TP * PP * CP)))
(( TRAIN_BATCH_SIZE % PPO_MINI_BATCH_SIZE == 0 )) || { echo "Batch must divide into PPO mini-batches" >&2; exit 2; }
(( PPO_MINI_BATCH_SIZE * ROLLOUT_N % (DP * MICRO_BATCH_SIZE) == 0 )) || { echo "Invalid PPO micro-batch division" >&2; exit 2; }
(( ROLLOUT_N >= 2 )) || { echo "GRPO needs multiple responses per prompt" >&2; exit 2; }

[[ -d "$VERL_ROOT/verl" ]] || { echo "VERL_ROOT not found: $VERL_ROOT" >&2; exit 2; }
# Vendor environment scripts may reference unset variables; restore nounset afterwards.
set +u
source /usr/local/Ascend/cann/set_env.sh
if [[ -f /usr/local/Ascend/nnal/atb/set_env.sh ]]; then
    source /usr/local/Ascend/nnal/atb/set_env.sh
fi
set -u
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_MAX_CONNECTIONS=1 VLLM_USE_V1=1 VLLM_ALLREDUCE_USE_SYMM_MEM=0
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
# Installed MindSpeed-Bridge autoload hooks provide the Qwen3.5 Ascend integration.
# No reference-script PYTHONPATH pointing into another user's checkout.
cd "$VERL_ROOT"

ARGS=(
    model_engine=megatron
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    algorithm.rollout_correction.bypass_mode=False
    "data.train_files=${TRAIN_FILE}"
    "data.val_files=${TEST_FILE}"
    "data.train_batch_size=${TRAIN_BATCH_SIZE}"
    "data.max_prompt_length=${MAX_PROMPT_LENGTH}"
    "data.max_response_length=${MAX_RESPONSE_LENGTH}"
    data.filter_overlong_prompts=True
    data.truncation=error
    "actor_rollout_ref.model.path=${MODEL_PATH}"
    actor_rollout_ref.model.trust_remote_code=False
    "actor_rollout_ref.model.use_remove_padding=${USE_REMOVE_PADDING}"
    actor_rollout_ref.model.mtp.enable=False
    actor_rollout_ref.model.mtp.enable_train=False
    actor_rollout_ref.model.mtp.enable_rollout=False
    "actor_rollout_ref.actor.optim.lr=${LR}"
    "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}"
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE}"
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${MAX_TOKEN_LEN}"
    "actor_rollout_ref.actor.use_dynamic_bsz=${USE_DYNAMIC_BSZ}"
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.01
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.megatron.use_mbridge=True
    "actor_rollout_ref.actor.megatron.use_remove_padding=${USE_REMOVE_PADDING}"
    "actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${TP}"
    "actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${PP}"
    "actor_rollout_ref.actor.megatron.context_parallel_size=${CP}"
    "actor_rollout_ref.actor.megatron.expert_model_parallel_size=${EP}"
    "actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${ETP}"
    actor_rollout_ref.actor.megatron.param_offload=True
    actor_rollout_ref.actor.megatron.optimizer_offload=True
    actor_rollout_ref.actor.megatron.dtype=bfloat16
    actor_rollout_ref.actor.checkpoint.strict=False
    actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=auto
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_aux_loss_coeff=0.01
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_z_loss_coeff=0.001
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type=alltoall
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_naive_l2norm=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_triton_gdn=False
    "+actor_rollout_ref.actor.megatron.override_transformer_config.use_ascend_gdn=${USE_ASCEND_GDN}"
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=False
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1.0
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True
    actor_rollout_ref.rollout.name=vllm
    "actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP}"
    "actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_MEMORY_UTIL}"
    "actor_rollout_ref.rollout.n=${ROLLOUT_N}"
    actor_rollout_ref.rollout.dtype=bfloat16
    "actor_rollout_ref.rollout.enforce_eager=${ENFORCE_EAGER}"
    "actor_rollout_ref.rollout.max_num_seqs=${MAX_NUM_SEQS}"
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${USE_DYNAMIC_BSZ}"
    "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${MAX_TOKEN_LEN}"
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.free_cache_engine=True
    +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=0
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    "actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${USE_DYNAMIC_BSZ}"
    "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${MAX_TOKEN_LEN}"
    "actor_rollout_ref.ref.megatron.param_offload=${REF_PARAM_OFFLOAD}"
    # Ref inherits actor's parallel mesh, remove-padding and transformer overrides.
    critic.enable=False
    trainer.device=npu
    'trainer.logger=[console]'
    trainer.project_name=verl_qwen35_geo3k_a5
    "trainer.experiment_name=${MODE}_${RUN_ID}"
    trainer.nnodes=1
    "trainer.n_gpus_per_node=${NDEVICES_PER_NODE}"
    trainer.val_before_train=False
    "trainer.save_freq=${SAVE_FREQ}"
    "trainer.test_freq=${TEST_FREQ}"
    trainer.total_epochs=15
    "trainer.total_training_steps=${TOTAL_STEPS}"
    trainer.resume_mode=disable
    "trainer.default_local_dir=${CKPT_DIR}"
    "ray_kwargs.ray_init.num_cpus=${RAY_NUM_CPUS}"
    +ray_kwargs.ray_init.address=local
    +ray_kwargs.ray_init.include_dashboard=False
    "+ray_kwargs.ray_init._temp_dir=${RAY_TMP_DIR}"
)

if [[ "$ACTION" == config ]]; then
    exec "$PYTHON" -m verl.trainer.main_ppo "${ARGS[@]}" "$@" --cfg job --resolve
fi

# Refuse an existing run directory so a same-second launch cannot overwrite logs.
mkdir -p -- "$(dirname -- "$RUN_DIR")"
mkdir -- "$RUN_DIR"
exec > >(tee "$RUN_DIR/launcher.log" "$RUN_DIR/train.log") 2>&1
{
    echo "===== Run environment ====="
    echo "Started: $(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S %Z (%z)')"
    echo "Python executable: $(command -v "$PYTHON")"
    "$PYTHON" --version
    "$PYTHON" -m pip --version
    echo "===== pip list ====="
    "$PYTHON" -m pip list --format=columns --disable-pip-version-check
    echo "===== pip show: core training dependencies ====="
    "$PYTHON" -m pip show torch torch-npu transformers vllm vllm-ascend \
        megatron-core mindspeed mindspeed-bridge ray memfabric-hybrid || \
        echo "WARNING: pip show could not report every requested package; see diagnostics above."
    echo "===== End run environment ====="
} 2>&1 | tee "$RUN_DIR/environment.txt"
echo "MODE=$MODE | NPU=$NDEVICES_PER_NODE TP=$TP PP=$PP CP=$CP EP=$EP ETP=$ETP DP=$DP | rollout TP=$GEN_TP"
echo "Output: $RUN_DIR"
echo "Resolving final Hydra configuration (no Ray startup)..."
"$PYTHON" -m verl.trainer.main_ppo "${ARGS[@]}" "$@" --cfg job --resolve > "$RUN_DIR/resolved-config.yaml"
cp -- "$SELF" "$RUN_DIR/launcher.sh"
git rev-parse HEAD > "$RUN_DIR/verl-commit.txt"
"$PYTHON" -m pip freeze > "$RUN_DIR/pip-freeze.txt"
export PREFLIGHT_CONFIG="$RUN_DIR/resolved-config.yaml"

echo "Preflight: files, NPU visibility, backend imports; no model weights loaded."
"$PYTHON" - <<'PY'
import importlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import sys
import traceback

errors = []
# Validate the resolved config, including any final user-supplied Hydra overrides.
from omegaconf import OmegaConf
cfg = OmegaConf.load(os.environ["PREFLIGHT_CONFIG"])
model_path = cfg.actor_rollout_ref.model.path
for package in ("torch", "torch-npu", "transformers", "vllm", "vllm-ascend", "megatron-core", "mindspeed", "mindspeed-bridge"):
    try:
        print(f"VERSION {package}={metadata.version(package)}", flush=True)
    except metadata.PackageNotFoundError:
        errors.append(f"Missing package: {package}")
try:
    model = Path(model_path)
    json.loads((model / "config.json").read_text())
    index = json.loads((model / "model.safetensors.index.json").read_text())
    shards = set(index["weight_map"].values())
    assert all((model / f).is_file() and (model / f).stat().st_size > 0 for f in shards), "Missing model shard"
    import pyarrow.parquet as pq
    for name, paths in (("train", cfg.data.train_files), ("test", cfg.data.val_files)):
        for path in ([paths] if isinstance(paths, str) else paths):
            data = pq.ParquetFile(path)
            assert data.metadata.num_rows > 0
            assert {"prompt", "images", "reward_model", "data_source"} <= set(data.schema_arrow.names)
            print(f"DATA {name}: {data.metadata.num_rows} rows ({path})", flush=True)
except Exception as exc:
    errors.append(f"Model/data check: {exc}")
try:
    import torch
    import torch_npu
    assert torch.npu.is_available(), "NPU unavailable"
    count = torch.npu.device_count()
    assert cfg.trainer.device == "npu" and cfg.trainer.nnodes == 1, "This launcher targets a single NPU node"
    assert count >= cfg.trainer.n_gpus_per_node, f"Only {count} visible NPUs"
    print(f"NPU visible count={count}", flush=True)
except Exception as exc:
    errors.append(f"NPU check: {exc}")
try:
    # MindSpeed adaptation must be installed before importing Megatron-Core.
    import mindspeed.megatron_adaptor
    from megatron.bridge import AutoBridge
    bridge = AutoBridge.from_hf_pretrained(model_path, trust_remote_code=cfg.actor_rollout_ref.model.trust_remote_code)
    from mindspeed_bridge.runtime.auto_register import get_status
    print("MindSpeed-Bridge:", get_status(), flush=True)
    if cfg.actor_rollout_ref.actor.megatron.override_transformer_config.use_ascend_gdn:
        importlib.import_module("fla_npu")
        importlib.import_module("mindspeed_bridge.models.qwen_vl.modelling_qwen3_vl.flash_gated_delta_rule")
except Exception as exc:
    errors.append(f"Megatron/Ascend GDN import: {exc}")
    traceback.print_exc(limit=4)
try:
    from vllm import LLM, SamplingParams
    importlib.import_module("vllm_ascend.vllm_ascend_C")
except Exception as exc:
    errors.append(f"vLLM import: {exc}")
    traceback.print_exc(limit=4)
if errors:
    print("\nPRECHECK FAILED; training and Ray have not been started:", flush=True)
    for error in errors:
        print(f"  - {error}", flush=True)
    sys.exit(1)
print("PRECHECK PASSED (imports only; full training remains unverified)", flush=True)
PY

if [[ "$ACTION" == check ]]; then
    echo "Check complete. No training started."
    exit 0
fi

echo "Starting $TOTAL_STEPS step(s). Model/data stay offline; checkpoints: $CKPT_DIR"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
"$PYTHON" -u -m verl.trainer.main_ppo "${ARGS[@]}" "$@"
