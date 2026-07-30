---
tags:
  - type/note
  - theme/machine-learning
aliases: ["Section 10 - K-Means Clustering"]
lead: K-means partitions data into K clusters by iteratively assigning points to the nearest centroid and updating centroids.
created: 2026-04-27
modified: 2026-04-28
source: "HackTheBox, Fundamentals of AI, Section 10."
---

![[k_means_clustering.png]]

`K-means clustering` partitions a dataset into `K` non-overlapping clusters by grouping data points that are close to each other in feature space. It minimizes within-cluster variance — points inside a cluster should be as similar as possible, while points in different clusters should be as dissimilar as possible.

The algorithm is iterative:

1. `Initialization`: Randomly select `K` data points as initial cluster centroids.
2. `Assignment`: Assign each data point to the nearest centroid using `Euclidean distance`.
3. `Update`: Recompute each centroid as the mean of all points assigned to its cluster.
4. `Iteration`: Repeat steps 2 and 3 until centroids stop changing significantly or a maximum number of iterations is reached.

## Euclidean Distance

The distance metric used in the assignment step is `Euclidean distance`:

$$d(\mathbf{x}, \mathbf{y}) = \sqrt{\sum_{i=1}^{n}(x_i - y_i)^2}$$

Where `x_i` and `y_i` are the values of the `i`-th feature for points `x` and `y`. Because this metric is scale-sensitive, features must be standardized before running K-means.

## Choosing the Optimal K

The number of clusters `K` must be specified in advance. There is no universal method for choosing it — domain knowledge, visual inspection, and quantitative heuristics all play a role.

### Elbow Method

![[k_means_elbow.png]]

The elbow method plots the within-cluster sum of squares (WCSS) against different values of `K`:

1. Run K-means for `K = 1, 2, 3, ...` up to some maximum.
2. Record the WCSS for each `K`. WCSS measures total squared distance of each point from its centroid.
3. Plot WCSS vs. `K`.
4. The "elbow" — where WCSS starts decreasing more slowly — suggests a good `K`.

Past the elbow, adding more clusters yields diminishing returns on variance reduction and risks overfitting to noise.

### Silhouette Analysis

![[silhouette_analysis.png]]

The silhouette score gives each data point a value between -1 and 1 based on how well it fits its assigned cluster relative to neighboring clusters:

- Score near **1**: point is well inside its cluster, far from others.
- Score near **0**: point sits near a cluster boundary.
- Score near **-1**: point is likely misassigned.

To use it for K selection:

1. Run K-means for a range of `K` values.
2. Compute the average silhouette score across all points for each `K`.
3. Choose the `K` with the highest average score.

### Domain Expertise and Other Considerations

Quantitative metrics help, but domain context often overrides them. A customer segmentation task constrained to three marketing teams calls for `K = 3` regardless of what the elbow plot suggests. Additional factors:

- `Computational Cost`: WCSS and silhouette scores both grow with `K`, so very large `K` values add overhead.
- `Interpretability`: Clusters need to be explainable. Dozens of clusters are hard to act on.

## Data Assumptions

- `Cluster Shape`: K-means assumes spherical clusters of roughly equal size. It fails on elongated, irregular, or highly unequal clusters.
- `Feature Scale`: Larger-scale features dominate distance calculations. Always standardize before applying K-means.
- `Outliers`: Centroids are means, making them sensitive to extreme values. Outliers can pull centroids away from true cluster centers.

---

## Summary

- K-means partitions a dataset into K clusters by iteratively assigning points to the nearest centroid (by Euclidean distance) and recomputing centroids as cluster means.
- The algorithm repeats the assign-update cycle until centroids stabilize or a maximum iteration count is reached.
- K must be specified in advance; the elbow method and silhouette analysis are quantitative tools for choosing it, but domain knowledge often overrides metrics.
- The elbow method plots WCSS vs. K — the "elbow" where reduction slows suggests a good K value.
- Silhouette scores range from -1 to 1; near 1 means well-clustered, near 0 means boundary point, near -1 means likely misassigned.
- K-means assumes spherical, similarly-sized clusters and is sensitive to outliers (which pull centroids away from true cluster centers) and feature scale.

---

## Best Practices

- Always standardize features before running K-means — Euclidean distance makes K-means highly sensitive to feature scale differences.
- Run K-means multiple times with different random initializations (K-means++) and select the run with the lowest WCSS to avoid poor local minima.
- Use the elbow method as a starting point and silhouette analysis to validate — check both rather than relying on a single heuristic.
- Let domain constraints guide K when they exist (e.g., three product categories → K=3), overriding quantitative suggestions.
- Remove or cap outliers before clustering — extreme values pull centroids and distort the resulting cluster assignments.
- Check the assumption of spherical clusters by plotting the data after dimensionality reduction; if shapes are elongated or irregular, consider DBSCAN or Gaussian mixture models instead.

---

## Quiz

**Q1:** Describe the four steps of the K-means algorithm.
> 1) Initialize K random centroids. 2) Assign each point to the nearest centroid by Euclidean distance. 3) Recompute each centroid as the mean of its assigned points. 4) Repeat steps 2-3 until centroids no longer change significantly.

**Q2:** What does the elbow method plot and how is the optimal K identified?
> It plots Within-Cluster Sum of Squares (WCSS) against values of K. As K increases WCSS decreases; the "elbow" — where the rate of decrease slows sharply — suggests a good K beyond which additional clusters yield diminishing variance reduction.

**Q3:** What does a silhouette score near -1 indicate for a data point?
> The point is likely misassigned — it is closer to a neighboring cluster's centroid than to its own cluster's centroid.

**Q4:** Why is K-means unsuitable for elongated or irregularly shaped clusters?
> K-means uses Euclidean distance to a centroid and assumes spherical, similarly-sized clusters. Elongated or non-convex shapes violate this assumption, causing the algorithm to split or merge clusters incorrectly.

---
# Back Matter

**Source**
- based_on:: [[HTB-COAE-Prep/1-Attacks/6-HTB-AI-Data-Attacks/0-Module-Information/Module-Description]]

**References**
- see:: [[Section-2-Mathematics-Refresher-for-AI]] — centroid distance uses Euclidean norm
- see:: [[Section-12-Anomaly-Detection]] — clustering can surface outlier clusters

**Terms**
- centroid, K selection, elbow method, silhouette score, inertia, cluster assignment, Lloyd algorithm
