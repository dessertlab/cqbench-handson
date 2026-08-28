# Lab setup — for the workshop organisers

**Session:** Benchmarking the Quality of AI Code Generators (hands-on, 3 hours)
**Repository:** <https://github.com/dessertlab/cqbench-handson>

## The short version

**No GPU is required.** Nothing is generated during the session — every code
sample the participants analyse already ships in the repository. The workshop
runs two static analyzers (Pylint and Semgrep) over 1,000 Python functions on
the **CPU**. A DGX or a graphics card will sit idle.

What each machine needs is modest: Miniconda, about 2.5 GB of free disk, and an
internet connection *once*, to build the environment.

## Per-machine requirements

| | |
|---|---|
| **Operating system** | macOS or Linux. **Native Windows will not work** — Semgrep has no Windows build. Windows machines need WSL2 (Ubuntu), where everything works unchanged. |
| **GPU** | Not used. |
| **CPU** | Any modern multi-core CPU. The full run takes ~100 s on 2 cores and less on more; 4+ cores is comfortable. |
| **RAM** | 4 GB is enough. |
| **Disk** | ~2.5 GB free per user: ~2 GB conda environment, ~25 MB repository. |
| **Python** | Installed by conda (3.11). Nothing needs to pre-exist. |
| **Network** | Needed **only** while creating the environment (~500 MB of packages). The workshop itself runs fully offline. |

## What we would like done before the school

Creating the conda environment takes 3–5 minutes per machine. With a room full
of participants doing it simultaneously over conference wifi, it can take much
longer and it eats into a three-hour session. **If the environment is already
present on each machine, the participants start working immediately.**

On each lab machine:

```bash
git clone https://github.com/dessertlab/cqbench-handson.git
cd cqbench-handson

conda env create -f environment.yml
conda activate cqbench-handson

python setup/verify_setup.py     # must print "Ready."
```

`verify_setup.py` checks both analyzer binaries and their exact versions, then
scores 8 tasks end to end. If it prints `Ready.`, that machine is done.

## If the lab machines have no internet

Build the environment once on a connected machine, then move it:

```bash
# on the connected machine
conda install -c conda-forge conda-pack
conda pack -n cqbench-handson -o cqbench-handson.tar.gz

# on each lab machine
mkdir -p ~/miniconda3/envs/cqbench-handson
tar -xzf cqbench-handson.tar.gz -C ~/miniconda3/envs/cqbench-handson
~/miniconda3/envs/cqbench-handson/bin/conda-unpack
```

The repository itself can be copied from a USB stick — it is ~25 MB and needs
no network.

## Why the analyzer versions are pinned

`environment.yml` pins `pylint==3.3.6` and `semgrep==1.120.0`. These are not
arbitrary: they are the versions the underlying study used, and a different
version reports different findings on the same code. Please do not upgrade them
— the participants' numbers would stop matching the published results, which is
one of the things the session teaches.

## Fallback if a machine fails

The repository ships `results/precomputed/`, the output of an identical run.
Every analysis notebook falls back to it automatically. A participant whose
setup fails loses only the live scoring step and can follow the whole session
otherwise — so a few failures are not a problem.

## Contact

Any questions before the school: *(add your contact)*
