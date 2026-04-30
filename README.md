# Spectral Clustering Project Bundle

This bundle contains a runnable Python workspace, a LaTeX report, generated figures, CSV results, and a compact status checklist.

## Directory layout

- `main.tex` - report source
- `refs.bib` - bibliography used by the report
- `final_pic/` - figures included by the report
- `results/` - CSV tables and intermediate summaries
- `code/` - Python workspace used to generate the figures and results
- `TODO.md` - concise status log of what was completed in this revision
- `PLAN.md` - lightweight extension notes

## Quick start

From the project root:

```bash
cd code
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_all.py
```

This regenerates the figures in `../final_pic/` and the CSV outputs in `../results/`.

To compile the paper:

```bash
cd ..
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## What the workspace implements

The codebase centers on normalized spectral clustering in the Ng-Jordan-Weiss style:

1. build an affinity graph from data,
2. form the symmetric normalized Laplacian,
3. compute the bottom eigenvectors,
4. row-normalize the spectral embedding,
5. cluster rows of the embedding with `k`-means.

The workspace also includes:

- synthetic geometric benchmarks,
- digit and Iris experiments,
- graph clustering on karate club and an SBM,
- noise and runtime scaling studies,
- additional experiments on parameter sensitivity, eigengaps, graph construction, graph fragility, and runtime bottlenecks.

## Notes for new contributors

- Figures in the report use the files under `final_pic/`. If you rename a figure, update both the plotting code and `main.tex`.
- Tables in the report are written manually from `results/*.csv`; when rerunning experiments, re-check the report numbers against the saved CSVs.
- The digits experiments use `sklearn.datasets.load_digits` for speed and reproducibility.
- The graph baseline is intentionally simple: `k`-means on adjacency rows. It is included as a transparent comparator, not as a state-of-the-art graph-clustering method.

## Suggested extension points

- repeated-seed robustness tables,
- local self-tuning graph scales,
- additional graph datasets,
- stronger graph-native baselines,
- automated report-table generation from CSV.
