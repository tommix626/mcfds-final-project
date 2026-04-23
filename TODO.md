# TODO / Status Audit Against `FinalProjects.pdf` (Spectral Clustering track)

This file compares the current bundle against the Spectral Clustering project brief and records what is now finished, what is only partially finished, and what is still left undone.

---

## Overall status

The bundle is now **substantially closer to a full project submission** than the earlier scaffold.

### Finished in a strong way
- One clear spectral clustering flavor is selected and implemented: **Ng--Jordan--Weiss**.
- The report now explains the **graph Laplacian**, the **normalized cut objective**, and the **eigenvector relaxation**.
- The algorithm section now states **inputs, outputs, steps, and computational cost**.
- The codebase is runnable from a single entrypoint: `code/run_all.py`.
- There are now **5 success-style synthetic experiments**, including **high-dimensional** cases.
- There are now **2 explicit failure/stress experiments**.
- The report compares results with **k-means** throughout the vector-data experiments.
- The exact digit progression requested in the PDF is now present:
  - `{0,1}`
  - `{1,7}`
  - `{0,1,3,6}`
  - all digits
- Cluster quality is now reported with **accuracy, ARI, NMI, and purity**.
- There is now a **noise robustness experiment**.
- There is now a **runtime/memory scaling experiment**.
- There is now stronger graph coverage via **karate club** and **stochastic block model**.

### Still only partial
- The project brief suggests a broad range of graph/community benchmarks. We cover two, but not the full variety.
- The computational discussion is much stronger than before, but still not a full theoretical complexity study of nearest-neighbor search.
- The report discusses expectations and whether they were met, but could still be made more exhaustive on a per-dataset basis.

### Still not finished
- No repeated-seed uncertainty analysis.
- No systematic parameter sweep over the graph-construction hyperparameters beyond the implemented fixed settings.
- No additional large benchmark datasets such as football / dolphins / political blogs.

---

## Requirement-by-requirement status

## 1) Theory part

### PDF asks
- Write a summary on spectral clustering.
- Pick one flavor.
- Describe its objective and how it constructs clusters from initial data.

### Status
**Finished.**

### What was added
- The report now includes:
  - graph construction and Laplacians,
  - the normalized cut objective,
  - the eigenvector relaxation idea,
  - a short comparison of flavors,
  - an explanation of why NJW was chosen.

### Desired result
Already achieved at a good project-report level.

---

## 2) Algorithmic summary

### PDF asks
- Summarize an algorithm of your choice.
- State clearly the inputs and outputs.
- Ideally discuss computational expense.

### Status
**Finished.**

### What was added
- The report now gives a step-by-step algorithm description.
- Inputs, outputs, and graph-construction choices are explicit.
- Cost is discussed both asymptotically and experimentally.

### Desired result
Already achieved.

---

## 3) Implementation requirement

### PDF asks
- Implement the algorithm.
- It should be easy to run.

### Status
**Finished.**

### What was added
- The codebase is modular.
- `run_all.py` regenerates all figures and CSV tables.
- The root README documents the workflow.

### Desired result
Already achieved.

---

## 4) At least 3 synthetic examples where the method should work

### PDF asks
- At least 3 synthetic examples.
- Qualitatively different.
- Some low-dimensional, some high-dimensional.

### Status
**Finished, and now stronger than the minimum.**

### Current coverage
- `two_moons`
- `circles`
- `blobs`
- `high_dim_moons`
- `high_dim_blobs`

### Desired result
Already achieved.

---

## 5) At least 1 or 2 synthetic examples where the method should not work well

### PDF asks
- 1 or 2 failure cases.
- Explain whether the outcomes match expectations.

### Status
**Finished.**

### Current coverage
- `overlapping_blobs_failure`
- `bridge_failure`

### Notes
The report now explicitly describes why these are stress cases and how the outcomes compare to expectation.

### Desired result
Already achieved.

---

## 6) Compare with k-means on all examples

### PDF asks
- For all examples, compare the results with k-means and comment on them.

### Status
**Mostly finished.**

### Finished part
- All vector-data experiments have a direct spectral-vs-k-means comparison.
- The report contains synthetic and digit summary tables.

### Partial part
- For graph datasets, plain k-means is not natural. The current bundle uses `kmeans_on_adjacency` as a transparent baseline.
- This is a reasonable comparison, but it is weaker than a graph-native baseline.

### What could still be improved
- Add one more graph-specific comparator, e.g. label propagation or modularity-based clustering.

### Desired result
A more defensible graph comparison section.

---

## 7) Meaningful performance measurement

### PDF asks
- Measure performance in a meaningful way.
- For digit subsets, discuss purity.

### Status
**Finished.**

### What was added
- Accuracy after best permutation matching
- ARI
- NMI
- Purity
- Dedicated digits-purity summary figure

### Desired result
Already achieved.

---

## 8) Specific digit subset sequence

### PDF asks
- Use the subsets:
  - all 1s and 0s
  - all 1s and 7s
  - all 1s, 3s, 6s, 0s
  - all digits
- Choose `K` accordingly.
- Measure how pure the clusters are.

### Status
**Finished.**

### What was added
- All four required subsets are now implemented and run.
- Purity is reported explicitly in the table and figure.

### Desired result
Already achieved.

---

## 9) Add noise and discuss the effect

### PDF asks
- Add Gaussian noise and discuss how results change.

### Status
**Finished.**

### What was added
- Two-moons noise sweep with ARI curves for spectral clustering and k-means.
- Report discussion now interprets the degradation.

### Desired result
Already achieved.

---

## 10) Quantitative computational / memory discussion

### PDF asks
- Discuss how computationally and memory intensive the algorithm is.
- At least experimentally / quantitatively.

### Status
**Finished at a practical level, partial at a deeper theoretical level.**

### Finished part
- Runtime is now broken into graph construction, eigensolver, and final k-means.
- Approximate memory usage is recorded.
- Scaling plots are generated.

### Still left undone
- No separate nearest-neighbor scaling study.
- No alternative sparse eigensolver comparison.

### Why left undone
These would significantly expand the scope and are not necessary for a solid project submission.

### Desired next result
A parameter sweep over `n_neighbors`, graph sparsity, and perhaps alternative eigensolver settings.

---

## 11) Breadth of real/graph datasets

### PDF asks or suggests
- Real datasets and graph examples such as karate, SBM, dolphins, football, etc.

### Status
**Partially finished.**

### Finished part
- Iris
- Digits subsets
- Karate club
- Stochastic block model

### Still left undone
- No dolphins / football / political blogs / LFR benchmark.

### Why left undone
The bundle was optimized to stay runnable offline and concise while still covering both vector and graph settings.

### Desired next result
Add one moderate-size real graph benchmark such as football or dolphins.

---

## 12) Repeated-seed stability / parameter sensitivity

### Status
**Not finished.**

### What is missing
- Re-running experiments across multiple random seeds.
- Error bars or mean/std tables.
- A sweep over `n_neighbors`.

### Sketch of how to proceed
1. Wrap each experiment in a loop over seeds, e.g. `seed in {0,1,2,3,4}`.
2. Aggregate ARI / purity / accuracy into mean and standard deviation.
3. For each dataset, sweep `n_neighbors` over a small grid and record the best / median result.
4. Add one figure for sensitivity to `n_neighbors` on two representative datasets.

### Desired result
A stronger report that distinguishes algorithm behavior from luck in initialization or graph choice.

---

## 13) Optional polish still worth doing

### Status
**Not necessary, but high-value if time remains.**

### Possible additions
- Add a graph-native comparator for karate/SBM.
- Add one more real graph dataset.
- Add confidence intervals over seeds.
- Add a small appendix table with all exact runtime and memory numbers.
- Add parameter-sensitivity figures for `n_neighbors`.

---

## Bottom line

If graded against the PDF in spirit, the project is now **close to submission quality** and clearly stronger than the original scaffold. The biggest remaining upgrades are:
1. repeated-seed robustness,
2. parameter sensitivity,
3. one more graph benchmark or graph baseline.
