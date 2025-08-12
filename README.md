# Setting up the environment

1. install `conda` using the distribution of your choice ([Anaconda](https://www.anaconda.com/download), [miniconda](/docs/getting-started/miniconda/install#windows-installation), [miniforge](https://github.com/conda-forge/miniforge), etc.).
All should work as long as you have access to the `conda` executable in a fresh terminal.
If you don't know what to choose, we recommend miniforge.

2. Create the environment to use for the semester.

```bash
conda create -n mpc2025 python=3.12 pip
conda activate mpc2025
pip install -r locked-requirements.txt
```

# Updating the environment

If the `locked-requirements.txt` changes, pull the latest version of this repo, activate the environment again and simply reinstall the dependencies:
```bash
git pull
conda activate mpc2025
pip install -r locked-requirements.txt
```

# Relocking the dependencies (for TAs)

If you have [`uv`](https://docs.astral.sh/uv/) installed locally you can run
```bash
uv pip compile requirements.txt -o locked-requirements.txt --emit-find-links --universal --python-version 3.12
```
if not, you can simply manually trigger the [`lock requirements` workflow](.github/workflows/lock_requirements.yml) on your branch, which will automatically relock the dependencies and push the changes if there are some.
