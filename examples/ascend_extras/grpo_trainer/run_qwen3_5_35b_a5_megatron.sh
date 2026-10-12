#!/usr/bin/env bash
# Qwen3.5 / Geo3K: reference batching preset on 8 x Ascend 950.
# Validated: 20 training steps with verl ee4ffacbaa08c24b9e1ace1317145255f0bdc0d1.
# Uses dynamic batching, remove-padding and backend-default gradient fusion.
# Layout: PROJECT_ROOT contains verl and Megatron-Bridge; model/data paths are overridable.
# Preserve dependencies, GDN isolation, workload, offload, and all run/profiling controls.
# Usage: bash this-script --print-config | --check | (no argument to train).
# PROFILE_ENABLE=False disables tracing. Final Hydra arguments override this preset.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BASE_LAUNCHER=${BASE_LAUNCHER:-${SCRIPT_DIR}/run_qwen3_5_35b_a5_base_megatron.sh}
ACTION_ARGS=()
case "${1:-}" in
    --print-config|--check) ACTION_ARGS=("$1"); shift ;;
    --help|-h) sed -n '2,7p' "${BASH_SOURCE[0]}"; exit 0 ;;
esac

export TZ=Asia/Shanghai
export CPU_AFFINITY_CONF=1
export MODE=train
export VERL_ROOT=${VERL_ROOT:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}
export PROJECT_ROOT=${PROJECT_ROOT:-$(dirname -- "$VERL_ROOT")}
export RUN_ID=${RUN_ID:-$(date +%Y%m%d-%H%M%S)-$$}
export RUN_DIR=${RUN_DIR:-${PROJECT_ROOT}/runs/qwen35_geo3k_${MODE}_${RUN_ID}}
export TENSORBOARD_DIR=${TENSORBOARD_DIR:-${RUN_DIR}/tensorboard}
# Use this environment's existing checkout, not the reference author's directory.
BRIDGE_SRC=${PROJECT_ROOT}/Megatron-Bridge/src
[[ -d "$BRIDGE_SRC/megatron/bridge" ]] || { echo "Missing installed Bridge checkout: $BRIDGE_SRC" >&2; exit 2; }
export PYTHONPATH="${BRIDGE_SRC}${PYTHONPATH:+:${PYTHONPATH}}"
# Current verl removed vanilla_mbridge: use_mbridge=True selects Megatron-Bridge.
# Current param_offload=True also releases gradient buffers; do not inject the
# removed grad_offload field. Keep MindSpeed-Bridge's installed autoload hooks.

# Match the supplied A5 NPU overrides; retain the baseline workload and mesh.
export NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-8}
export TP=${TP:-2} PP=${PP:-1} CP=${CP:-1} EP=${EP:-8} ETP=${ETP:-1}
export GEN_TP=${GEN_TP:-8}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
export ROLLOUT_N=${ROLLOUT_N:-5}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
export MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
export MAX_TOKEN_LEN=${MAX_TOKEN_LEN:-4096}
export USE_DYNAMIC_BSZ=${USE_DYNAMIC_BSZ:-True}
export USE_REMOVE_PADDING=${USE_REMOVE_PADDING:-True}
export ROLLOUT_MEMORY_UTIL=${ROLLOUT_MEMORY_UTIL:-0.65}
# Reference omits these options; pin this checkout's defaults for reproducibility.
export MAX_NUM_SEQS=${MAX_NUM_SEQS:-256}
export TOTAL_STEPS=${TOTAL_STEPS:-20}
export SAVE_FREQ=${SAVE_FREQ:--1} TEST_FREQ=${TEST_FREQ:--1}

# Retain reference offload; use this checkout's default graph-mode setting.
export REF_PARAM_OFFLOAD=${REF_PARAM_OFFLOAD:-True}
export ENFORCE_EAGER=${ENFORCE_EAGER:-False}
export RAY_NUM_CPUS=${RAY_NUM_CPUS:-64}

PROFILE_ENABLE=${PROFILE_ENABLE:-True}
PROFILE_STEPS=${PROFILE_STEPS:-[20]}
PROFILE_RANKS=${PROFILE_RANKS:-[0]}
PROFILE_ALL_RANKS=${PROFILE_ALL_RANKS:-False}
PROFILE_PATH=${PROFILE_PATH:-${RUN_DIR}/profiler}
EXTRA_ARGS=(
    # The base script forces False; delete that override to match the reference's
    # unspecified fusion setting. Actor/ref inherit the installed provider default.
    '~actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion'
    'trainer.logger=[console,tensorboard]'
    actor_rollout_ref.actor.ppo_epochs=1
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.enable_prefix_caching=True
    +actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.rl_config.enabled=True
)
case "$PROFILE_ENABLE" in
    True|true|1)
        EXTRA_ARGS+=(
            global_profiler.tool=npu
            "global_profiler.steps=${PROFILE_STEPS}"
            global_profiler.profile_continuous_steps=False
            "global_profiler.save_path=${PROFILE_PATH}"
        )
        for role in actor ref rollout; do
            EXTRA_ARGS+=(
                "actor_rollout_ref.${role}.profiler.enable=True"
                "actor_rollout_ref.${role}.profiler.all_ranks=${PROFILE_ALL_RANKS}"
                "actor_rollout_ref.${role}.profiler.ranks=${PROFILE_RANKS}"
                "actor_rollout_ref.${role}.profiler.tool_config.npu.contents=[cpu,npu]"
                "actor_rollout_ref.${role}.profiler.tool_config.npu.level=level1"
                "actor_rollout_ref.${role}.profiler.tool_config.npu.analysis=False"
            )
        done
        # Actor/ref: capture the whole selected step, including offload gaps.
        # vLLM requires discrete=True for its start_profile/stop_profile API.
        EXTRA_ARGS+=(
            actor_rollout_ref.actor.profiler.tool_config.npu.discrete=False
            actor_rollout_ref.ref.profiler.tool_config.npu.discrete=False
            actor_rollout_ref.rollout.profiler.tool_config.npu.discrete=True
        )
        ;;
    False|false|0) ;;
    *) echo "PROFILE_ENABLE must be True or False" >&2; exit 2 ;;
esac

# Do not create a run for --print-config. Preserve both launchers for real runs/checks.
if [[ ${ACTION_ARGS[0]:-} != --print-config ]]; then
    [[ ! -e "$RUN_DIR" ]] || { echo "Run directory already exists: $RUN_DIR" >&2; exit 2; }
    # The base launcher owns directory creation; the wrapper is copied after it exits.
    trap 'if [[ -d "$RUN_DIR" ]]; then cp -- "${BASH_SOURCE[0]}" "$RUN_DIR/perf-launcher.sh"; fi' EXIT
fi
bash "$BASE_LAUNCHER" "${ACTION_ARGS[@]}" "${EXTRA_ARGS[@]}" "$@"
