# TODO / Status

## Completed in this revision

- removed assignment-style meta language from the report
- added a dedicated evaluation-metrics subsection with definitions and equations for accuracy-after-matching, ARI, NMI, and purity
- revised figure captions so they state whether axes are original coordinates or PCA coordinates
- added a bridge-case discussion explaining why a visually plausible 2D cut can still have only moderate ARI
- renamed Section 9 to `Additional experiments`
- removed the meta framing paragraph that previously introduced Section 9
- added a `k`-means reference column to the parameter-sensitivity heatmap
- added a companion figure with representative row and column slices of the sensitivity map
- expanded the graph-construction subsection with explicit affinity equations for dense RBF, symmetric `k`-NN RBF, and mutual-`k`-NN RBF graphs
- reframed the old failure subsection as graph fragility rather than universal failure
- updated README so a new contributor can run the code and compile the paper quickly
- refreshed the bibliography with verified core references

## Still worth doing next

- repeat the sensitivity and noise studies over multiple random seeds, then plot mean and standard deviation bands
- add one self-tuning local-scale experiment following Zelnik-Manor and Perona
- add a stronger graph-native baseline for graph datasets, such as modularity maximization or a standard community-detection heuristic
- generate the LaTeX tables directly from CSV so the report numbers never drift from the saved outputs
- add confusion matrices for the digit experiments after optimal label matching

## Why these items remain open

These are all worthwhile, but they were left for a later pass because they are expansions rather than fixes. The current bundle already has a coherent paper, runnable code, and the extra experiments requested for the current revision.
