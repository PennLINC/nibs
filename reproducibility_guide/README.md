# MIRROR reproducibility guide

This directory is a self-contained [Jupyter Book](https://jupyterbook.org/)
for reproducing the MIRROR manuscript workflow after source-scalar generation.

## Build locally

From the repository root:

```bash
python -m venv /tmp/mirror-guide-env
source /tmp/mirror-guide-env/bin/activate
python -m pip install -r reproducibility_guide/requirements.txt
jupyter-book build reproducibility_guide
```

Open `reproducibility_guide/_build/html/index.html` in a browser. The generated
`_build/` directory is ignored by `reproducibility_guide/.gitignore`.

## Publish

The book is ready for a standard Jupyter Book GitHub Pages workflow. Publishing
requires a workflow under `.github/workflows/`; it is intentionally not added
as part of this guide-only change.

