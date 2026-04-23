# TODO: Gap analysis against `FinalProjects.pdf` (Spectral Clustering track)

This document compares the current bundle against the exact Spectral Clustering project brief and lists what is still missing or only partially covered.

## Executive summary

The current version is **good as a runnable scaffold**, but it is **not yet fully compliant** with the PDF if interpreted literally. It already covers:
- one clear flavor of spectral clustering (Ng–Jordan–Weiss),
- a theory section with objective and construction,
- a runnable Python implementation,
- several synthetic experiments,
- comparison to k-means,
- one graph dataset (karate club),
- one tabular dataset (iris),
- one noise experiment.

However, the PDF asks for a few things that are either missing or only weakly satisfied:
- a more explicit and deeper theory summary tied to the project references,
- a clearer quantitative discussion of computational and memory cost,
- synthetic experiments that are more clearly split into **success cases** and **failure cases**, with explicit commentary,
- the specific MNIST-style sequence of subsets (0/1, 1/7, 0/1/3/6, all digits),
- a more explicit analysis of cluster purity on those digit subsets,
- stronger coverage of graph-style benchmarks beyond karate club,
- stronger evidence that the experiments match expectations.

---

## Requirement-by-requirement comparison

## 1) Theory part

### PDF asks
- Write a summary on spectral clustering.
- Pick one flavor.
- Describe its **objective** and how it constructs clusters from initial data.

### Current status
**Mostly covered, but could be stronger.**
- `main.tex` does pick one flavor: Ng–Jordan–Weiss.
- It describes graph construction, normalized Laplacian, eigenvectors, row normalization, and k-means.
- It also mentions the normalized cut viewpoint.

### What is still missing
1. The writeup does **not yet explicitly derive or cleanly formalize** the optimization objective / relaxation.
2. The writeup does **not deeply compare flavors** enough to justify why this flavor was selected.
3. The writeup does not use the specific suggested references from the PDF very fully.

### How to proceed
- Add a dedicated subsection:
  - `Unnormalized cut / ratio cut / normalized cut`
  - `Relaxation to eigenvectors of the Laplacian`
  - `Why NJW uses row-normalized eigenvectors + k-means`
- Add a concise comparison paragraph:
  - Shi–Malik normalized cuts
  - Ng–Jordan–Weiss algorithmic variant
  - Laplacian eigenmaps relation
- Expand the bibliography and citations around these points.

### Desired result
A theory section that reads like a real project report rather than a scaffold: mathematically precise, reference-backed, and clearly tied to the chosen algorithm.

---

## 2) Algorithmic part: clear algorithm summary

### PDF asks
- Summarize an algorithm of your choice.
- State clearly the **inputs** and **outputs**.
- Ideally discuss how expensive it is computationally.

### Current status
**Covered at a basic level.**
- Inputs, outputs, and steps are present in `main.tex`.
- Complexity is discussed at a high level.

### What is still missing
1. Complexity is **not tied to measured experiments**.
2. Memory usage is only described informally.
3. There is no ablation showing effect of `n`, `d`, `k`, or graph sparsity on runtime.

### How to proceed
- Add a benchmark script, e.g. `benchmark_scaling.py`, that varies:
  - number of points `n`,
  - ambient dimension `d`,
  - number of neighbors `m`,
  - number of requested clusters `k`.
- Record:
  - graph construction time,
  - eigensolver time,
  - k-means time,
  - peak memory or approximate sparse/dense memory footprint.
- Produce plots:
  - runtime vs `n`,
  - runtime breakdown by stage,
  - memory vs `n`.

### Desired result
A section with both asymptotic discussion and concrete measurements. The PDF explicitly asks for computational and memory discussion, at least quantitatively or experimentally.

---

## 3) Implementation requirement

### PDF asks
- Implement the algorithm.
- It should be easy for the instructor or anyone else to run.

### Current status
**Covered.**
- The codebase is organized and runnable.
- `README.md` gives setup and execution steps.
- Python is fine here because the handoff explicitly asked for Python.

### What is still missing
Nothing essential here, but two quality upgrades would help:
1. Add CLI flags for selecting datasets and parameters.
2. Add deterministic seed handling and repeated-seed evaluation.

### How to proceed
- Replace `run_all.py` with argparse support.
- Add `configs/` or a small YAML/JSON config system.

### Desired result
A cleaner handoff for Claude Code and easier reproducibility.

---

## 4) Synthetic datasets where the method should work

### PDF asks
- At least **3 synthetic examples**, qualitatively different from each other.
- Some low-dimensional, some high-dimensional.
- Cases where the algorithm is expected to work well.

### Current status
**Partially covered.**
- Current success examples: `two_moons`, `circles`, `blobs`.
- These are all standard and good.

### What is still missing
1. They are mostly **low-dimensional toy datasets**.
2. There is not yet a clearly designated **high-dimensional synthetic success case**.
3. The writeup does not explicitly mark these as “expected success cases” and explain why beforehand.

### How to proceed
Add at least one high-dimensional synthetic benchmark, for example:
- `high_dim_sbm_features`: generate 2–4 clusters in 50D or 100D with anisotropic noise but graph-separable neighborhoods.
- `high_dim_two_moons_embedding`: embed moons into 50D by appending noise coordinates.

For each success case:
- state expectation before showing results,
- report metrics,
- compare against k-means,
- explain why spectral should succeed.

### Desired result
At least four success examples, with at least one being genuinely high-dimensional.

---

## 5) Synthetic datasets where the method should not work well

### PDF asks
- At least **1 or 2 synthetic examples**, qualitatively different from each other, where you expect the algorithms to not work well.
- Ask whether the results match expectations, yes/no and why.

### Current status
**Only partially covered.**
- There is one weak failure case: `overlapping_blobs_failure`.

### What is still missing
1. Only one failure example is present.
2. It is not qualitatively very different from the success cases.
3. The report does not explicitly say “this is expected to fail because ...” and then evaluate whether that expectation was met.

### How to proceed
Add at least one more failure regime, ideally very different in character:
- `varying_density_moons`: same manifold class but highly uneven densities, which can break fixed-kNN choices.
- `bridged_clusters`: two blobs connected by a thin bridge, producing ambiguity in graph cuts.
- `xor_checkerboard`: a case where local neighborhoods do not align well with the target partition.
- `noisy_high_dim_blobs`: high-dimensional noise swamps neighborhood structure.

For each failure case:
- state expected failure mode,
- run spectral and k-means,
- explain exactly what broke.

### Desired result
Two clearly distinct failure examples, each with interpretation.

---

## 6) Compare with k-means on all examples

### PDF asks
- For all examples, compare with k-means and comment on results.

### Current status
**Mostly covered, but not fully.**
- Synthetic datasets: yes.
- Iris: yes.
- Digits subset: yes.
- Karate graph: no direct k-means baseline, since it is treated as a graph-only case.

### What is still missing
1. The narrative comments are brief.
2. The karate example does not have a natural k-means baseline in the current report.
3. The comparison is not yet summarized in a single master table in the LaTeX report.

### How to proceed
- Add one summary table to `main.tex` listing all datasets and both methods.
- For graph datasets, either:
  - explain why plain feature-space k-means is not directly comparable, or
  - compare against a graph baseline such as modularity or label propagation instead.
- Expand commentary: one paragraph per dataset family.

### Desired result
A report where the comparison with k-means is impossible to miss and easy to grade.

---

## 7) Meaningful performance metrics

### PDF asks
- Measure performance using a meaningful notion of performance.
- For digit subsets, explicitly look at how “pure” the clusters are.

### Current status
**Mostly covered, but digits purity coverage is incomplete.**
- ARI, NMI, purity, and accuracy are already computed.
- That is good.

### What is still missing
1. Purity is computed, but the report does not foreground it enough for the digit tasks.
2. There is no confusion-matrix or cluster-composition table for digits.
3. There is no per-cluster purity discussion.

### How to proceed
- Add cluster composition tables for each digits subset.
- Add per-cluster purity plots or a stacked bar chart.
- In `main.tex`, explicitly connect the metrics to the PDF wording about “pure” clusters.

### Desired result
A grader can immediately see that the project answered the “how pure are the clusters?” part directly.

---

## 8) MNIST-style digits experiments

### PDF asks
Perform spectral clustering on the following subsets of MNIST digits data:
- all 1s and 0s,
- all 1s and 7s,
- all 1s, 3s, 6s, 0s,
- all digits,
and use `K` equal to the number of classes in each subset.
If the data are too large, extract a subset and **indicate clearly when and how**.

### Current status
**Not fully covered.**
- Current code only runs **one** digits experiment: classes `{0,1,3,6}`.
- It uses a PCA-2D version for reliability and speed.

### What is still missing
1. Missing the `0 vs 1` subset.
2. Missing the `1 vs 7` subset.
3. Missing the `all digits` subset.
4. The report does not clearly explain the subsampling rule in a way that mirrors the PDF request.
5. The current digits run uses PCA-2D for the actual experiment, while the report says clustering is done in original standardized feature space; this should be checked carefully and made consistent.

### How to proceed
Implement a dedicated digits pipeline:
- `run_digits_subset(classes=[0,1], n_per_class=...)`
- `run_digits_subset(classes=[1,7], n_per_class=...)`
- `run_digits_subset(classes=[0,1,3,6], n_per_class=...)`
- `run_digits_subset(classes=list(range(10)), n_per_class=...)`

Also:
- decide whether to use sklearn digits or MNIST;
- if using sklearn digits as a substitute, say so explicitly;
- if using MNIST, add a download script and cache.

For each subset:
- set `K` correctly,
- report ARI/NMI/purity,
- compare to k-means,
- show a compact purity/confusion summary.

### Desired result
A direct one-to-one match to the PDF’s named digit experiments.

---

## 9) Add noise and discuss how results change

### PDF asks
- Add Gaussian noise with different standard deviations.
- Discuss how results change.

### Current status
**Covered, but lightly.**
- The current bundle does a noise sweep on two moons.

### What is still missing
1. Only one dataset is tested under noise.
2. The discussion is short.
3. No repeated-seed confidence intervals are shown.

### How to proceed
- Extend the noise sweep to:
  - moons,
  - circles,
  - one digits subset.
- For each noise level, run multiple seeds.
- Plot mean ± std for ARI or purity.

### Desired result
A stronger noise robustness section rather than a single illustrative figure.

---

## 10) Use of graph datasets / community detection

### PDF framing
The project is about spectral clustering for both:
- community detection in graphs,
- clustering high-dimensional data.

### Current status
**Partially covered.**
- High-dimensional / feature data: yes.
- Graph community detection: only karate club.

### What is still missing
1. Only one small graph benchmark is included.
2. No stochastic block model experiment, even though the PDF explicitly suggests it.
3. No degree-heterogeneous graph benchmark like LFR.

### How to proceed
Add:
- `run_sbm_graphs()` for planted communities with tunable separation,
- `run_lfr_graphs()` for harder degree-heterogeneous community detection,
- optionally dolphins or football if lightweight data loading is easy.

For each graph benchmark:
- show graph visualization,
- report ARI/NMI vs ground truth,
- discuss where spectral works or fails.

### Desired result
Balanced coverage of both graph clustering and feature clustering.

---

## 11) Commentary on whether results matched expectations

### PDF asks
- “Do the results of running the algorithm match your expectations? Yes/no and why?”

### Current status
**Partially covered.**
- The report says the results mostly align with standard intuition.

### What is still missing
1. The yes/no expectation check is not done **dataset by dataset**.
2. Failure cases are not analyzed deeply enough.

### How to proceed
For each dataset in the final report, add a short block:
- **Expectation:** ...
- **Observed:** ...
- **Match?** Yes/No
- **Reason:** ...

### Desired result
A report that visibly answers the project prompt rather than assuming the reader will infer it.

---

## 12) Real-data breadth

### PDF suggests
Examples include iris and MNIST-style tasks, alongside graph datasets.

### Current status
**Acceptable but still thin.**
- Iris: yes.
- Digits subset: yes.
- Karate: yes.

### What is still missing
A stronger sense of breadth. One more real benchmark would help.

### How to proceed
Choose one lightweight addition:
- full sklearn digits subsets as above,
- dolphins / football network,
- a small text or image embedding dataset if already available.

### Desired result
A project that feels complete rather than minimal.

---

## Priority order

## Highest priority
1. **Finish the exact digits sequence from the PDF**: `0/1`, `1/7`, `0/1/3/6`, `all digits`.
2. **Add at least one more failure case** and analyze it explicitly.
3. **Add quantitative runtime/memory experiments**.
4. **Strengthen the theory section** with objective/relaxation details.

## Medium priority
5. Add SBM and/or LFR graph experiments.
6. Add repeated-seed robustness for noise and metrics.
7. Add master result tables and per-dataset expectation/observation blocks.

## Nice-to-have
8. Add ablations over graph construction choices.
9. Add sparse eigensolver benchmarking.
10. Add appendix derivations and more visualizations.

---

## Concrete code tasks

### Code tasks to add
- `code/spectral_project/benchmarks.py`
- `code/spectral_project/digits_suite.py`
- `code/spectral_project/graph_benchmarks.py`
- `code/spectral_project/ablations.py`

### Figures/tables to add
- `final_pic/runtime_vs_n.png`
- `final_pic/runtime_breakdown.png`
- `final_pic/memory_vs_n.png`
- `final_pic/digits_01_*.png`
- `final_pic/digits_17_*.png`
- `final_pic/digits_all_*.png`
- `final_pic/failure_case_2_*.png`
- `final_pic/sbm_graph_*.png`
- `results/digits_suite_metrics.csv`
- `results/runtime_metrics.csv`
- `results/graph_metrics.csv`
- `results/cluster_purity_tables.csv`

---

## Bottom line

The current bundle is a **strong starter package**, but it is still closer to a **well-built scaffold** than a fully complete final submission under a strict reading of the PDF. The biggest gap is that the current project does **not yet fully execute the exact digit experiments requested in the PDF**, and it does **not yet quantitatively discuss computational/memory cost** in the experimental sense.

If the goal is “good enough to hand to Claude Code for completion,” this bundle is in solid shape. If the goal is “touch every point in the project brief,” the items above should still be completed.
