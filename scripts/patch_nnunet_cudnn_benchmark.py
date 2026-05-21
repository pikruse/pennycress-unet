#!/usr/bin/env python
"""Patch nnU-Net to make cudnn.benchmark controllable by environment.

nnU-Net v2 enables torch.backends.cudnn.benchmark in its training entrypoint.
On ROCm this can trigger expensive MIOpen find/search during startup. The patch
is intentionally small and idempotent so the Slurm launcher can apply it after
the conda environment is activated.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


OLD = "cudnn.benchmark = True"
NEW = (
    'cudnn.benchmark = os.environ.get("NNUNET_CUDNN_BENCHMARK", "0").lower() '
    'in ("1", "true", "t", "yes", "y")'
)


def main() -> None:
    spec = importlib.util.find_spec("nnunetv2")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("Could not find installed nnunetv2 package")

    package_dir = Path(next(iter(spec.submodule_search_locations))).resolve()
    path = package_dir / "run" / "run_training.py"
    text = path.read_text()

    if NEW in text:
        print(f"nnU-Net cudnn.benchmark patch already present: {path}")
        return

    count = text.count(OLD)
    if count == 0:
        raise RuntimeError(
            f"Could not find expected assignment {OLD!r} in {path}. "
            "nnU-Net may have changed; inspect run_training.py manually."
        )

    backup = path.with_suffix(path.suffix + ".pennycress.bak")
    if not backup.exists():
        backup.write_text(text)

    path.write_text(text.replace(OLD, NEW))
    print(f"Patched {count} cudnn.benchmark assignment(s) in {path}")


if __name__ == "__main__":
    main()
