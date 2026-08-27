#!/usr/bin/env python3
"""Execute notebooks top to bottom and report the first failure in each.

    python tools/run_notebooks.py            # all of notebooks/
    python tools/run_notebooks.py 03 04      # only matching ones

Used to check the session materials actually run before handing them out.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

REPO = Path(__file__).resolve().parents[1]


def run(path: Path) -> bool:
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(notebook, timeout=1800, kernel_name="python3",
                            resources={"metadata": {"path": str(path.parent)}})
    started = time.time()
    try:
        client.execute()
    except CellExecutionError as exc:
        print(f"FAIL  {path.relative_to(REPO)}  ({time.time() - started:.0f}s)")
        print("      " + str(exc).strip().splitlines()[-1][:300])
        for line in str(exc).strip().splitlines():
            if "Error" in line or "error" in line:
                print("      " + line[:300])
        return False
    print(f"ok    {path.relative_to(REPO)}  ({time.time() - started:.0f}s)")
    return True


def main(argv: list[str]) -> int:
    wanted = argv[1:]
    paths = sorted((REPO / "notebooks").rglob("*.ipynb"))
    paths = [p for p in paths if ".ipynb_checkpoints" not in str(p)]
    if wanted:
        paths = [p for p in paths if any(w in p.stem for w in wanted)]
    return 0 if all([run(p) for p in paths]) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
