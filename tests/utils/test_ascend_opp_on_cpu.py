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

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def isolation():
    path = Path(__file__).parents[2] / "verl/utils/ascend_opp.py"
    spec = importlib.util.spec_from_file_location("ascend_opp_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def config(role="actor_rollout_ref", enabled=True):
    return {
        "role": role,
        "config": {
            "actor": {
                "strategy": "megatron",
                "megatron": {"override_transformer_config": {"use_ascend_gdn": enabled}},
            },
            "rollout": {"name": "vllm"},
        },
    }


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    fla = tmp_path / "fla_npu"
    vendor = fla / "opp/vendors/fla_npu_transformer"
    opapi = vendor / "op_api/lib/libcust_opapi.so"
    opapi.parent.mkdir(parents=True)
    opapi.write_bytes(b"test placeholder")
    vllm = tmp_path / "vllm_ascend"
    vllm_vendor = vllm / "_cann_ops_custom/vendors/custom_transformer"
    vllm_vendor.mkdir(parents=True)
    torch = ModuleType("torch")
    torch.npu = SimpleNamespace(is_initialized=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", torch)
    platforms = ModuleType("vllm.platforms")

    def bootstrap():
        os.environ["ASCEND_CUSTOM_OPP_PATH"] = str(vllm_vendor) + ":" + os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")

    platforms.current_platform = SimpleNamespace(import_kernels=bootstrap)
    monkeypatch.setitem(sys.modules, "vllm.platforms", platforms)
    monkeypatch.setenv("ASCEND_CUSTOM_OPP_PATH", "/other/vendor")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/unchanged/lib")
    monkeypatch.delenv("FLA_NPU_OP_API_LIB", raising=False)
    return SimpleNamespace(
        fla=fla, vendor=vendor, opapi=opapi, vllm=vllm, vllm_vendor=vllm_vendor, torch=torch, platforms=platforms
    )


def bind_packages(isolation, monkeypatch, runtime):
    packages = {"fla_npu": runtime.fla, "vllm_ascend": runtime.vllm}
    monkeypatch.setattr(isolation, "_package_dir", packages.__getitem__)


@pytest.mark.parametrize("role", ["rollout", "critic", "rm"])
def test_non_training_roles_do_not_change_environment(isolation, monkeypatch, role):
    monkeypatch.setenv("ASCEND_CUSTOM_OPP_PATH", "/vllm/vendor:/other/vendor")
    isolation.prepare_fla_npu_training([config(role=role)])
    assert os.environ["ASCEND_CUSTOM_OPP_PATH"] == "/vllm/vendor:/other/vendor"


@pytest.mark.parametrize("enabled", [False, "False"])
def test_disabled_gdn_does_not_import_optional_backends(isolation, monkeypatch, enabled):
    monkeypatch.setattr(isolation, "_package_dir", lambda _: pytest.fail("unexpected package discovery"))
    isolation.prepare_fla_npu_training([config(enabled=enabled)])


def test_gpu_does_not_require_fla(isolation, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", ModuleType("torch"))
    monkeypatch.setattr(isolation, "_package_dir", lambda _: pytest.fail("unexpected package discovery"))
    isolation.prepare_fla_npu_training([config()])


def test_platform_prepend_is_removed_and_other_vendors_preserved(isolation, monkeypatch, runtime):
    bind_packages(isolation, monkeypatch, runtime)
    alias = runtime.vllm.parent / "vendor_alias"
    alias.symlink_to(runtime.vllm_vendor, target_is_directory=True)
    monkeypatch.setenv("ASCEND_CUSTOM_OPP_PATH", f"{alias}:{runtime.vllm_vendor}/op_api/lib:/other/vendor")
    isolation.prepare_fla_npu_training([config()])
    assert os.environ["ASCEND_CUSTOM_OPP_PATH"].split(":") == [
        str(runtime.vendor),
        str(runtime.vendor / "op_api/lib"),
        "/other/vendor",
    ]
    assert os.environ["FLA_NPU_OP_API_LIB"] == str(runtime.opapi)
    assert os.environ["LD_LIBRARY_PATH"] == "/unchanged/lib"
    assert runtime.vllm_vendor.is_dir()  # No shared-directory rename.
    isolation.prepare_fla_npu_training([config()])
    assert len(os.environ["ASCEND_CUSTOM_OPP_PATH"].split(":")) == 3
    # A separate native rollout process can restore its own vendor priority.
    runtime.platforms.current_platform.import_kernels()
    assert os.environ["ASCEND_CUSTOM_OPP_PATH"].split(":")[0] == str(runtime.vllm_vendor)


def test_late_isolation_fails_without_claiming_success(isolation, monkeypatch, runtime):
    runtime.torch.npu.is_initialized = lambda: True
    with pytest.raises(RuntimeError, match="before NPU initialization"):
        isolation.prepare_fla_npu_training([config()])
    assert os.environ["ASCEND_CUSTOM_OPP_PATH"] == "/other/vendor"


def test_missing_wheel_vendor_fails_before_platform_bootstrap(isolation, monkeypatch, runtime):
    bind_packages(isolation, monkeypatch, runtime)
    runtime.opapi.unlink()
    with pytest.raises(RuntimeError, match="wheel-embedded"):
        isolation.prepare_fla_npu_training([config()])
    assert os.environ["ASCEND_CUSTOM_OPP_PATH"] == "/other/vendor"


def test_platform_must_not_initialize_npu(isolation, monkeypatch, runtime):
    bind_packages(isolation, monkeypatch, runtime)

    def initialize_too_early():
        runtime.torch.npu.is_initialized = lambda: True

    runtime.platforms.current_platform.import_kernels = initialize_too_early
    with pytest.raises(RuntimeError, match="platform discovery initialized"):
        isolation.prepare_fla_npu_training([config()])


def test_other_backend_with_null_megatron_config(isolation, monkeypatch):
    kwargs = config()
    kwargs["config"]["actor"] = {"strategy": "fsdp2", "megatron": None}
    monkeypatch.setattr(isolation, "_package_dir", lambda _: pytest.fail("unexpected package discovery"))
    isolation.prepare_fla_npu_training([kwargs])
