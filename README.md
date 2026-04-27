# Spectral Clustering Final Project Workspace

This workspace contains a reproducible Python codebase, generated figures, CSV results, and an Overleaf-ready LaTeX report for the **Spectral Clustering** final project.

The report follows the project brief's structure:

1. theory summary,
2. algorithmic description and computational cost,
3. data exploration on synthetic, real/vector, and graph datasets,
4. comparison with k-means,
5. noise robustness,
6. additional research-style experiments that make the report stand out.

## Repository layout

```text
.
├── main.tex                  # Overleaf-ready report
├── refs.bib                  # Bibliography
├── README.md                 # This onboarding document
├── TODO.md                   # Concise experiment/status checklist
├── PLAN.md                   # Earlier planning notes and extension ideas
├── final_pic/                # Figures referenced by main.tex
├── results/                  # CSV metrics and JSON summaries
└── code/
    ├── requirements.txt
    ├── run_all.py            # One-command experiment runner
    └── spectral_project/
        ├── clustering.py     # Spectral clustering implementation
        ├── data.py           # Dataset construction/loading
        ├── experiments.py    # Experiment orchestration
        ├── metrics.py        # Accuracy/ARI/NMI/purity
        ├── plots.py          # Plotting utilities
        └── utils.py
```

## Quick start

Create an environment and install the dependencies:

```bash
cd code
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run every experiment and regenerate all figures/tables:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python run_all.py --random-state 0
```

Outputs are written to:

```text
final_pic/
results/
```

`main.tex` already references the expected figure filenames under `final_pic/`.

## Overleaf workflow

Upload the following to Overleaf:

```text
main.tex
refs.bib
final_pic/
```

Then compile with pdfLaTeX. No external data downloads are required; the generated figures are already included in the bundle.

## Experiment inventory

The current code runs:

- synthetic success/failure benchmarks: moons, circles, blobs, high-dimensional moons/blobs, overlapping blobs, bridge failure;
- real/vector benchmarks: Iris and `sklearn.datasets.load_digits`;
- required digit subset progression: `{0,1}`, `{1,7}`, `{0,1,3,6}`, and all digits;
- graph benchmarks: Zachary karate club and stochastic block model;
- Gaussian-noise robustness on two moons;
- runtime and approximate memory scaling;
- five research-style additions:
  - parameter sensitivity heatmap over noise and kNN graph degree,
  - eigengap study across datasets,
  - graph construction ablation,
  - failure mode taxonomy,
  - runtime bottleneck breakdown.


## Recent revision: figure-caption consistency

The current version includes a pass that aligns captions and discussion with the actual plots and metric tables. In particular:

- high-dimensional moons are described as a modest quantitative advantage, not a clean visual recovery;
- the digit PCA figure is described as illustrative only, with the quantitative table carrying the claim;
- the noise curve is described as single-seed and non-monotone after graph disruption;
- the bridge example is treated as a stress/partial-degradation case rather than a catastrophic failure;
- the runtime bottleneck plot now overlays the plain k-means baseline used in the text.

This makes the report more cautious and defensible: visual figures illustrate behavior, while stronger claims are tied to the metric tables.

## Notes for new contributors

The main entrypoint is `spectral_project/experiments.py`. To add a new experiment:

1. add data generation/loading in `data.py` if needed;
2. add a runner function in `experiments.py`;
3. save metrics with `_save_results`;
4. save figures under `figures_dir()`;
5. add the runner to `run_all_experiments`;
6. reference the new figure/table in `main.tex`.

Keep experiments small enough to run on a laptop. The current workspace intentionally uses offline, built-in datasets so that the project remains easy to reproduce.
