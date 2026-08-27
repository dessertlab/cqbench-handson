#!/usr/bin/env python3
"""Turn the plain-Python sources in src_notebooks/ into .ipynb files.

Cells are delimited exactly as in the jupytext "light" format, so the sources
stay diffable and reviewable in git:

    # %% [markdown]
    # # A heading
    # some prose

    # %%
    print("a code cell")

    python tools/build_notebooks.py                 # build everything
    python tools/build_notebooks.py 03_rq1          # build one
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SOURCES = REPO / "src_notebooks"


def split_cells(text: str):
    cells, kind, buffer = [], "code", []

    def flush():
        source = "\n".join(buffer).strip("\n")
        if source.strip():
            cells.append((kind, source))
        buffer.clear()

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# %%"):
            flush()
            kind = "markdown" if "[markdown]" in stripped else "code"
            continue
        if kind == "markdown":
            buffer.append(line[2:] if line.startswith("# ") else
                          ("" if stripped == "#" else line))
        else:
            buffer.append(line)
    flush()
    return cells


def build(path: Path, out_dir: Path) -> Path:
    cells = []
    for index, (kind, source) in enumerate(split_cells(path.read_text(encoding="utf-8"))):
        cell = {"cell_type": kind, "id": f"{path.stem}-{index:03d}", "metadata": {},
                "source": source.splitlines(keepends=True)}
        if kind == "code":
            cell |= {"execution_count": None, "outputs": []}
        cells.append(cell)
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3 (cqbench-handson)",
                           "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / (path.stem + ".ipynb")
    target.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    return target


def main(argv: list[str]) -> int:
    wanted = argv[1:]
    built = []
    for source in sorted(SOURCES.rglob("*.py")):
        stem = source.stem
        if wanted and not any(w in stem for w in wanted):
            continue
        out = REPO / "notebooks"
        if source.parent.name == "solutions":
            out = out / "solutions"
        built.append(build(source, out))
    for path in built:
        print(f"built {path.relative_to(REPO)}")
    return 0 if built else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
