#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import runpy
import sys
from pathlib import Path


def patch_dataloader_workers():
    value = os.environ.get("AUTOBOTS_NUM_WORKERS")
    if value is None or value == "":
        return

    num_workers = int(value)
    import torch.utils.data

    original_dataloader = torch.utils.data.DataLoader

    def dataloader_with_env_workers(*args, **kwargs):
        kwargs["num_workers"] = num_workers
        if num_workers == 0:
            kwargs.pop("prefetch_factor", None)
            kwargs["persistent_workers"] = False
        return original_dataloader(*args, **kwargs)

    torch.utils.data.DataLoader = dataloader_with_env_workers


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: run_autobots_with_worker_patch.py /path/to/AutoBots/script.py [args...]")

    script = str(Path(sys.argv[1]).resolve())
    sys.argv = [script] + sys.argv[2:]
    sys.path.insert(0, str(Path(script).parent))
    patch_dataloader_workers()
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
