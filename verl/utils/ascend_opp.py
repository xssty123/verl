# Copyright 2026 verl contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Keep FLA training operators separate from vLLM-Ascend's OPP vendor."""

import importlib.util
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)
_TRAINING_ROLES = {"actor", "ref", "actor_rollout", "actor_rollout_ref"}


def _uses_fla_with_vllm(kwargs: dict) -> bool:
    if kwargs.get("role") not in _TRAINING_ROLES:
        return False
    config = kwargs.get("config")
    if config is None:
        return False
    actor = config.get("actor") or {}
    rollout = config.get("rollout") or {}
    if actor.get("strategy") != "megatron" or rollout.get("name") != "vllm":
        return False
    megatron = actor.get("megatron") or {}
    overrides = megatron.get("override_transformer_config") or {}
    return str(overrides.get("use_ascend_gdn", False)).lower() == "true"


def _package_dir(name: str) -> Path:
    spec = importlib.util.find_spec(name)
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError(f"FLA training OPP isolation requires an installed {name} package.")
    return Path(next(iter(spec.submodule_search_locations))).resolve()


def prepare_fla_npu_training(worker_kwargs: list[dict]) -> None:
    """Prepare a colocated training process before Worker.__init__ sets its device.

    vLLM's platform discovery can prepend its custom OPP even in training
    workers. Its same-named GDN operators need not have FLA's internal attribute
    layout. Finish that discovery, then remove only vLLM's vendor from this
    process. Native rollout servers initialize their own vLLM platform in
    separate processes; no shared operator directories are changed.
    """
    if not any(_uses_fla_with_vllm(kwargs) for kwargs in worker_kwargs):
        return

    import torch

    npu = getattr(torch, "npu", None)
    if npu is None:
        return
    if npu.is_initialized():
        raise RuntimeError("FLA training OPP isolation must run before NPU initialization; start a fresh worker.")

    fla_vendor = _package_dir("fla_npu") / "opp" / "vendors" / "fla_npu_transformer"
    fla_opapi = fla_vendor / "op_api" / "lib" / "libcust_opapi.so"
    vllm_ops = _package_dir("vllm_ascend") / "_cann_ops_custom"
    if not fla_vendor.is_dir() or not fla_opapi.is_file():
        raise RuntimeError(f"FLA training OPP isolation requires the wheel-embedded vendor at {fla_vendor}.")

    from vllm.platforms import current_platform

    current_platform.import_kernels()
    if npu.is_initialized():
        raise RuntimeError("vLLM platform discovery initialized the NPU before FLA OPP isolation.")

    entries = [p for p in os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(os.pathsep) if p]
    retained = [p for p in entries if not Path(p).resolve().is_relative_to(vllm_ops)]
    # Use the package-local OPP before runtime initialization, not an unrelated
    # toolkit vendor or an op_api library selected through global symbol lookup.
    paths = [str(fla_vendor), str(fla_vendor / "op_api" / "lib"), *retained]
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = os.pathsep.join(dict.fromkeys(paths))
    os.environ["FLA_NPU_OP_API_LIB"] = str(fla_opapi)
    logger.warning("Prepared FLA training OPP before NPU initialization: %s", fla_vendor)
