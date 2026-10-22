#!/usr/bin/env bash
set -euo pipefail
# Entry migration only; experiment settings and Python body are preserved.
exec "${TMM_PYTHON:-/data0/pretrained/env_humanomniv2/bin/python}" -B - "$0" "$@" <<'PYCODE'
import os as _entry_os
import sys as _entry_sys
__file__ = _entry_os.path.abspath(_entry_sys.argv[1])
_entry_sys.argv = [_entry_sys.argv[1], *_entry_sys.argv[2:]]
from pathlib import Path
import sys

sys.dont_write_bytecode = True
phi_dependencies = Path('/data0/pretrained/tmm_phi4_deps')
if phi_dependencies.is_dir():
    sys.path.insert(0, str(phi_dependencies))

# 修改这里后运行：bash scripts/evidence_interventions.sh
SETTINGS = {
    "gpu": 3,                   # GPU 编号
    # QO=Qwen2.5-Omni-7B, PM=Phi-4-multimodal
    "models": None,  # None 跑全表
    "tasks": None,              # None 跑全表；指定任务如 ["daic"]
    "check": False,             # True 只检查资源
    "stage": "run",
    "seeds": [27],              # 冻结模型确定性推理；三种子仅用于 controller 训练
    "config": "configs/experiments.json",
}

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from tmm.support.experiments import main

if __name__ == "__main__":
    raise SystemExit(main(4, SETTINGS))
PYCODE
