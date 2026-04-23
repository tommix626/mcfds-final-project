# Spectral Clustering Project Bundle

This bundle contains:
- `main.tex`: LaTeX report.
- `refs.bib`: bibliography.
- `final_pic/`: figures referenced by the report.
- `code/`: runnable Python workspace.

## Quick start

```bash
cd code
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python run_all.py
```

That command regenerates all figures into `../final_pic/` and tables into `../results/`.

## Overleaf
Upload the contents of this bundle to Overleaf and compile `main.tex`.
