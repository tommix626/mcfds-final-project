# TODO / Status Checklist

This file summarizes what has been completed and what remains optional.

## Completed core project requirements

- [x] Theory summary for normalized spectral clustering.
- [x] Explanation of graph construction, normalized Laplacian, normalized cut motivation, and NJW-style row-normalized embedding.
- [x] Runnable Python implementation of normalized spectral clustering.
- [x] k-means baseline throughout.
- [x] Synthetic success examples: moons, circles, blobs, high-dimensional moons, high-dimensional blobs.
- [x] Synthetic stress/failure examples: overlapping blobs and bridge-connected clusters.
- [x] Real/vector examples: Iris and handwritten digits.
- [x] Required digit progression: `{0,1}`, `{1,7}`, `{0,1,3,6}`, and all digits.
- [x] Purity/accuracy/ARI/NMI metrics.
- [x] Noise robustness experiment.
- [x] Runtime and approximate memory scaling.
- [x] Graph/community examples: Zachary karate club and stochastic block model.
- [x] Overleaf-ready `main.tex` with figure references.
- [x] One-command runner: `cd code && python run_all.py`.

## Completed stand-out additions

- [x] Parameter sensitivity map over noise and kNN graph degree.
  - Output: `final_pic/parameter_sensitivity_heatmap.png`
  - Metrics: `results/parameter_sensitivity_metrics.csv`

- [x] Eigengap study across easy, hard, digit, and graph datasets.
  - Output: `final_pic/eigengap_study.png`
  - Metrics: `results/eigengap_metrics.csv`

- [x] Graph construction ablation: kNN RBF vs mutual-kNN RBF vs dense RBF.
  - Output: `final_pic/graph_construction_ablation.png`
  - Metrics: `results/graph_construction_ablation.csv`

- [x] Failure mode taxonomy: graph fragmentation, bridge connection, varying-density geometry.
  - Output: `final_pic/failure_taxonomy.png`
  - Metrics: `results/failure_taxonomy_metrics.csv`

- [x] Runtime bottleneck breakdown.
  - Output: `final_pic/runtime_bottleneck_breakdown.png`
  - Metrics: `results/runtime_bottleneck_percent.csv`


## Caption/text consistency audit completed

- [x] Revised high-dimensional moons language.
  - Change: downgraded the claim from a visually clean recovery to a modest quantitative advantage.
  - Reason: the plot shows both methods are degraded after adding noisy coordinates.

- [x] Revised digit-subset discussion.
  - Change: emphasized that the PCA visualization is illustrative only and that the quantitative table is the stronger evidence.
  - Reason: a 2D PCA projection can hide differences that appear in the full metric table.

- [x] Revised noise-robustness language.
  - Change: described the curve as unstable/non-monotone after graph disruption rather than as smooth degradation.
  - Reason: the single-seed sweep has a small intermediate bump.

- [x] Revised bridge example.
  - Change: renamed it from a catastrophic failure to a stress/partial-degradation case.
  - Reason: the visual plot still separates the two main masses reasonably.

- [x] Revised runtime bottleneck figure.
  - Change: overlaid the plain k-means baseline on the bottleneck plot.
  - Reason: the text compares spectral runtime to k-means, so the figure should display the comparison directly.

## Optional remaining improvements

- [ ] Repeated-seed robustness.
  - Why left undone: it multiplies runtime and adds many numbers without changing the main narrative.
  - How to proceed: run each experiment over seeds `0..9`, report mean ± standard deviation for ARI/purity.

- [ ] Confusion matrices for digit subsets.
  - Why left undone: current digit section already includes purity, accuracy, ARI, NMI, and a visualization.
  - How to proceed: add optimal label matching, then plot confusion matrices for `{1,7}` and all digits.

- [ ] A stronger graph baseline.
  - Why left undone: the project explicitly asks for k-means comparison; adjacency-row k-means is included as a transparent baseline.
  - How to proceed: add modularity maximization or Louvain/Leiden if extra dependencies are allowed.

- [ ] Real MNIST instead of `sklearn.load_digits`.
  - Why left undone: `load_digits` is offline, small, and reproducible.
  - How to proceed: add an optional downloader for MNIST or OpenML and subsample to laptop-friendly size.

- [ ] Automated LaTeX table generation.
  - Why left undone: current report hardcodes small tables for readability.
  - How to proceed: write a script that exports CSV summaries to `.tex` table snippets.
