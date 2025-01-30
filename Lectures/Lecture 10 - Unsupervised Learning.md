---
tags:
  - Lectures
  - Marinus
  - _FirstPass
  - LLM
Created: 2025-01-09
---
# Unsupervised Learning

## Supervised Learning
- **Definition**: "Learning with labels" – dataset consists of pairs $(x_i, y_i)$ and a model $f$.
- **Goal**: Predict $\hat{y}_i = f(x_i)$ as closely as possible to the true labels $y_i$.
- **Loss Function**: Measures the closeness of predictions to the true labels, denoted as $L$.

## Unsupervised Learning
- **Definition**: No labels, only datapoints $x_i$.
- **Goal**: Find structure in the data and use a model $f$ for tasks like:
    - Dimensionality reduction
    - Clustering
    - Generating new data
    - Estimating density

---

# Dimensionality Reduction

## Definition
- **Dimensionality Reduction**: Reducing the number of features in a dataset while retaining as much information as possible.
- Transformation:
  $$
  X = 
  \begin{pmatrix}
  x_{11} & x_{12} & \dots & x_{1d} \\
  x_{21} & x_{22} & \dots & x_{2d} \\
  \vdots & \vdots & \ddots & \vdots \\
  x_{n1} & x_{n2} & \dots & x_{nd}
  \end{pmatrix}
  \rightarrow
  \bar{X} =
  \begin{pmatrix}
  \bar{x}_{11} & \dots & \bar{x}_{1r} \\
  \bar{x}_{21} & \dots & \bar{x}_{2r} \\
  \vdots & \ddots & \vdots \\
  \bar{x}_{n1} & \dots & \bar{x}_{nr}
  \end{pmatrix}
  $$
  where $r \ll d$.

---

## Motivation for Dimensionality Reduction
1. **Data Visualization**: Simplifies visualization by reducing dimensions to 2D or 3D.
2. **Data Preprocessing**: Reduces complexity for easier pre-processing of datasets with high dimensions.
3. **Computational Tractability**: Lowering dimensionality makes high-dimensional computations feasible.

---

# Curse of Dimensionality

- **Problem**: Algorithms and intuitions do not scale well with increasing dimensions.
- **Key Challenges**:
    1. **Data Sparsity**: Exponential increase in space volume causes data points to become sparse.
    2. **Combinatorics**: With $d$ binary variables, the number of combinations is $2^d$, which becomes unmanageable at high $d$.
- when a linear sepereation line is not enough, you might want to increase the dimensionality of the data..
- when you increase the dimensions, your datapoints that actaully convey the underlying pattern become much more sparse 
- this causes a mass underusage of space, and exponential growth of  that underusage. 
- hence the curse of dimensionality, it helps but only to a certain degree. 
- keep dimensionality low to convey as much information with as minimal performance cost as possible



## Sampling
- **Relationship**: $n \propto d$
  - Feature dimensionality defines the number of data points ($n$) that must be captured for a representative dataset.
  - As the number of features ($d$) increases, the required number of data points also increases exponentially.
- **Implication**: More features require larger datasets to ensure representation.

---

## Defining Curse of Dimensionality
### Intuitions
- **Low-Dimensional Intuitions**: Concepts and intuitions developed for low-dimensional spaces fail in higher-dimensional contexts.

### Performance
- **Impact on ML Models**: As dimensionality increases, the performance of machine learning models tends to decrease (features, spaces, etc.).

---

## Curse of Dimensionality – Sparsity
- **Example**: Sampling $n=9$ points into a unit hypercube.
    - As dimensionality $d$ increases, data points become sparser.
    - Representation for dimensions $d=1$ to $d=3$:
        - $d=1$: Divided into $3^1$ bins.
        - $d=2$: Divided into $3^2$ bins.
        - $d=3$: Divided into $3^3$ bins.

![[Pasted image 20250109151322.png]]

## Catcurse of Dogmensionality
![[Pasted image 20250109151614.png]]
- this can be seen as overfitting
- essentially, we dont want to make a category for every type of cat and every type of dog, and what doesnt fit in that SPECIFIC category the model wont be able to classify. 
- therefore, you dont want to add more dimensions than needed, you only want to classify cats and dogs, not some bs noise a high dimensionality unusable classification.

![[Pasted image 20250109151839.png]]
- more dimensions arent always better

# Curse of Dimensionality – Sampling

## Sampling in High Dimensions
- **Problem**: Sampling points within a given distribution becomes harder as the number of dimensions ($d$) increases.
- **Objective**: Sampling points inside a unit sphere:
    - Sample points inside a unit hypercube.
    - Retain only those points where $\|x\|_2 \leq 1$.
- **Observation**:
    - The proportion of points within the unit sphere decreases as $d$ increases.
    - Example table:
        | $d$ | Proportion Inside Sphere |
        |-----|---------------------------|
        | 1   | 1.00                      |
        | 2   | 0.78                      |
        | 3   | 0.52                      |
        | 10  | 0.03                      |
        | 100 | 0.00                      |

[Insert example scatter plots for 2D and 3D spheres vs. hypercubes.]

---

# Curse of Dimensionality – Cubes vs. Spheres

- **Volume Comparison**:
    - Volume of a $d$-dimensional hypersphere relative to a unit hypercube:
    $$
    V(d) = \frac{\pi^{d/2}}{\Gamma(d/2+1)} \cdot 0.5^d
    $$
    - As $d$ increases, the ratio of the sphere's volume to the cube's volume approaches zero.
- **Implication**:
    - Higher dimensions exacerbate sparsity, making most data points concentrate near boundaries.

[Insert plot of $V_{\text{sphere}} / V_{\text{cube}}$ vs. $d$ to illustrate volume ratio decreasing.]

---

# Dimensionality Reduction

## Objective
- Reduce dimensionality to:
    - Improve classification accuracy.
    - Choose an optimal set of features with lower dimensionality.

### Methods
1. **Feature Extraction**:
    - Generates new features from existing ones using a mapping function $f$ (linear or non-linear).
    - Example:
    $$
    X \xrightarrow{f(x)} Y, \quad K \ll N
    $$
2. **Feature Selection**:
    - Chooses a subset of the original features.
    - Example:
    $$
    X \xrightarrow{\text{selection}} Y, \quad K \ll N
    $$

# Principal Component Analysis (PCA)

## Overview
- **PCA**: A classic method for dimensionality reduction.
- **Concept**: Find a "rototranslation" (rotation + translation) of axes (dimensions) to capture the maximum variance in the data.

---

## PCA Steps

### 1. Translation (Centering Data)
- Center the data with respect to its mean:
  $$
  \bar{X} = X - \mathbb{E}[X]
  $$

### 2. Compute Covariance Matrix
- Covariance matrix $\Sigma$ captures how the data spreads and the interactions between features:
  $$
  \Sigma = \text{COV}(\bar{X}) = \bar{X}^T \bar{X}
  $$
  Example covariance matrix:
  $$
  \Sigma = 
  \begin{pmatrix}
  \sigma_{x_1}^2 & \sigma_{x_1, x_2} & \cdots & \sigma_{x_1, x_d} \\
  \sigma_{x_2, x_1} & \sigma_{x_2}^2 & \cdots & \sigma_{x_2, x_d} \\
  \vdots & \vdots & \ddots & \vdots \\
  \sigma_{x_d, x_1} & \sigma_{x_d, x_2} & \cdots & \sigma_{x_d}^2
  \end{pmatrix}
  $$

---

## Covariance in PCA
- Example covariance for a 2D dataset:
  $$
  \Sigma =
  \begin{pmatrix}
  5 & 1 \\
  1 & 1
  \end{pmatrix}
  $$

---

## Matrices as Transformations
- **Matrix Interpretation**: Transformations in space:
  $$
  \mathbf{w} = A\mathbf{v}, \quad A \in \mathbb{R}^{p \times q}
  $$
- Square matrices $A \in \mathbb{R}^{d \times d}$:
  - Perform isomorphism (rotation, reflection).
  - Example: Covariance matrix $\Sigma$ serves as a transformation.
![[Pasted image 20250109152656.png]]

---

## Eigen decomposition in PCA
- **Eigenvectors and Eigenvalues**:
  - **Eigenvectors**: Directions of maximum variance.
  - **Eigenvalues**: Magnitudes of variance.
  - Equation:
    $$
    \Sigma \mathbf{v} = \lambda \mathbf{v}
    $$
- **Decomposition**:
  $$
  \Sigma = V \Lambda V^T
  $$
  Where:
  - $V$ is the matrix of eigenvectors.
  - $\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_d)$ is the diagonal matrix of eigenvalues.

---

## PCA Recipe
1. **Eigendecompose** the covariance matrix:
   $$
   \Sigma = V \Lambda V^T
   $$
2. **Project the data** onto the eigenvectors:
   $$
   Y = \bar{X}V
   $$
   - $Y$: The transformed representation of the data.

[Insert diagram visualizing projection onto principal components.]

# Principal Component Analysis (PCA) – Dimensionality Reduction

## Eigenvalues and Dimensionality
- **Eigenvalues**: Represent the variance captured by each eigenvector.
    - Equivalent to the "magnitude" of pull in the geometric interpretation.
- **Dimensionality Reduction**:
    - Select the top $r$ eigenvectors and eigenvalues to create a reduced representation:
    $$
    Y = \bar{X} V_{[:,1:r]}
    $$

---

## PCA in Practice
- **Eigenvalue Spectrum** (Elbow Plot):
    - Visualizes the explained variance ratio (cumulative sum of eigenvalues).
    - Use the plot to determine the optimal number of components to retain.
    - Example: Scree plot where the "elbow" indicates the best number of components.

[Insert elbow plot showing variance explained.]

---

## PCA – Reprojection
- **Reprojecting Data**:
    - PCA allows reprojecting the reduced data back into the original space by inverting the transformation:
    $$
    \bar{X} = Y V^T + \mathbb{E}[X]
    $$
- **Reprojection Error**:
    - PCA minimizes the reprojection error:
    $$
    L = \|X - V^T X V\|^2
    $$

---

## PCA – Drawbacks
1. **Non-linear Variance**:
    - PCA does not retain non-linear variance.
    - Example: PCA cannot accurately represent shapes like 3D loops in 2D space.
2. **Global Variance**:
    - PCA focuses on global variance, which may overlook local patterns.
    - Solution: Techniques like t-SNE focus on preserving local variance.

[Insert visual examples comparing PCA and non-linear methods.]

---
# t-Stochastic Neighbor Embedding (t-SNE)

## Overview
- **t-SNE**: A popular non-linear dimensionality reduction method tailored for 2D/3D visualization.
- **Process**:
    1. Transforms distances in the input space into a probability distribution.
    2. Defines another probability distribution in the low-dimensional space.
    3. Optimizes to minimize the **KL divergence** between the two distributions.
    4. Samples the low-dimensional distribution to obtain the embedding.

---

## Key Characteristics
- **Stochastic**: Incorporates randomness in the algorithm, resulting in slightly different outputs on repeated runs.
- **Neighbor**: Focuses on retaining the variance of neighbor points.
- **Embedding**: Projects data into a lower-dimensional space.
- **Iterative Procedure**: Projection is performed in multiple steps to improve embedding.

---

## Probability Computation
1. **Neighborhood Probabilities**:
    - Use a Gaussian kernel to calculate probabilities in the high-dimensional space:
    $$
    p_{j|i} = \frac{\exp(-\|x_i - x_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-\|x_i - x_k\|^2 / 2\sigma_i^2)}, \quad \forall i \neq j
    $$
    - Intuition: Measure distance as the probability that point $x_j$ is within a Gaussian centered at $x_i$.
2. **Symmetrized Probabilities**:
    - Adjust for non-symmetric $p_{j|i}$ and $p_{i|j}$:
    $$
    p_{ij} = \frac{p_{j|i} + p_{i|j}}{2n}
    $$
    - $p_{ij}$ represents the similarity score (probability) between $x_i$ and $x_j$.

---

## Low-Dimensional Mapping
- Project data onto a low-dimensional space $\mathcal{Y}$ using a **t-distribution kernel**:
$$
q_{ij} = \frac{(1 + \|y_i - y_j\|^2)^{-1}}{\sum_{k \neq l}(1 + \|y_k - y_l\|^2)^{-1}}
$$
- **t-Distribution**: Provides robustness to outliers.

---

## KL Divergence and Optimization
- **KL Divergence**:
    - Measures the difference between the high-dimensional distribution $P$ and the low-dimensional distribution $Q$:
    $$
    KL(P \| Q) = \sum_{i,j|i \neq j} p_{ij} \log\frac{p_{ij}}{q_{ij}}
    $$
- **Training**:
    - Minimize $KL(P \| Q)$ via **gradient descent**.
    - Interpretation: Neighbors attract while non-neighbors repel.

---

## Hyperparameters
1. **Number of Iterations**:
    - Specifies how many gradient descent steps the algorithm will run.
2. **Perplexity**:
    - Related to the variance $\sigma_i^2$ of the Gaussian kernel for each point $x_i$.
    - Higher perplexity increases the neighborhood size.

---

## Practical Example
- **Visualization**:
    - Example: t-SNE applied to MNIST dataset (digits) with **perplexity=30**.
![[Pasted image 20250109153549.png]]

# Multi-Dimensional Scaling (MDS)

## Overview
- **MDS**: A dimensionality reduction technique similar to t-SNE but not stochastic.
- **Key Differences from t-SNE**:
    - t-SNE does not aim to preserve distances in the high-dimensional space.
    - MDS approximates distances from the high-dimensional space to the low-dimensional manifold.

---

## Methodology
1. **Pairwise Distance Calculation**:
   - Compute pairwise distances between points $x$ in the high-dimensional input space:
     $$
     d_{ij} = \|x_i - x_j\|
     $$

2. **Optimization**:
   - Learn a low-dimensional embedding $y$ that approximates the pairwise distances:
     $$
     L(y) = \sum_{i \neq j} (d_{ij} - \|y_i - y_j\|)^2
     $$
   - Use gradient descent to minimize $L(y)$.

3. **Result**:
   - The optimized embedding $y$ corresponds to a representation in lower dimensions, often preserving the pairwise distances approximately.

---

## Comparison: t-SNE vs MDS
- **MDS**:
    - Approximates distances in the high-dimensional space.
    - Results in embeddings with a low-dimensional manifold.
    - Generally faster but may not handle complex non-linear relationships as well as t-SNE.
- **t-SNE**:
    - Focuses on retaining the structure of neighbor points.
    - Often better for visualizing non-linear structures.

![[Pasted image 20250109153803.png]]

---
# Clustering

## What is Clustering?
- **Definition**: Clustering is an unsupervised learning technique that groups data points into clusters based on similarity.
- **Key Idea**: 
    - Elements in a cluster $C_i$ are more similar to each other than to elements in another cluster $C_j$.
    - Requires a measure of **similarity** among elements in a dataset.

![Clustering Example](Insert images showing raw data vs. clustered data.)

---

## Applications of Clustering
1. **User/Customer Segmentation**:
    - Identify and group users or customers based on behavior or characteristics.
2. **Community Detection**:
    - Identify communities in social networks.
3. **Medical Imaging**:
    - Classify different types of tissue.
4. **Recommendation Systems**:
    - Cluster similar items for personalized recommendations.
5. **Anomaly Detection**:
    - Identify unusual patterns in datasets.
6. **Image Segmentation**:
    - Partition images into meaningful segments.
7. **Natural Language Processing (NLP)**:
    - Handle word ambiguity or lexical grouping.

---

## Metrics for Clustering

### 1. **Within Cluster Sum of Squared Errors (WSS)**
- **Definition**: Computes the sum of squared distances from each point to its cluster centroid:
  $$
  WSS = \sum_{j=1}^C \sum_{i} d(c_j, x_i)^2
  $$
  - $d$: Distance metric (e.g., Euclidean distance).
  - $c_j$: Cluster centroid.

### 2. **Silhouette Value**
- **Calculation**:
  - $a(i)$: Mean distance between point $i$ and all other points in the same cluster:
    $$
    a(i) = \frac{1}{|C_i| - 1} \sum_{j \in C_i, j \neq i} d(x_i, x_j)
    $$
  - $b(i)$: Smallest mean distance from point $i$ to points in other clusters:
    $$
    b(i) = \min_{k \neq i} \sum_{j \in C_k} d(x_i, x_j)
    $$
  - Silhouette value for point $i$:
    $$
    s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}
    $$
  - Silhouette coefficient for the entire dataset:
    $$
    SC(X) = \frac{\sum_i s(i)}{n}
    $$

- **Interpretation**:
  - $s(i) \approx 1$: Well-clustered point.
  - $s(i) \approx 0$: Point on the border of two clusters.
  - $s(i) \approx -1$: Misclustered point.

---

## Selecting the Number of Clusters
1. **Elbow Method**:
    - Plot clustering metric (e.g., WSS) vs. $k$.
    - Choose $k$ where diminishing returns occur (elbow point).
2. **Silhouette Method**:
    - Evaluate $k$ values using the Silhouette coefficient.
    - Select $k$ with the highest coefficient.

[Insert diagrams showing examples of WSS and Silhouette methods.]
# Curse of Dimensionality
- **Sampling:** Number of data points $n \propto d$ (dimensions), exponentially increasing with dimensionality. More features require more samples for a representative dataset.
- **Challenges:**
  - **Intuitions:** Mathematical intuitions for low-dimensional spaces fail in high dimensions.
  - **Performance:** Machine learning models perform poorly with increased dimensionality.
- **Sparsity:** As dimensionality $d$ increases, data becomes sparser (e.g., distributing points in unit hypercubes).

# Dimensionality Reduction
- **Objective:** Select an optimal feature set to improve classification accuracy.
- **Methods:**
  - **Feature Extraction:** Derive new features from existing ones using linear or non-linear transformations.
  - **Feature Selection:** Choose a subset of original features.

# Principal Component Analysis (PCA)
- **Overview:** A classic dimensionality reduction method to maximize variance through rototranslation of axes.
- **Steps:**
  1. Center data: $\bar{X} = X - \mathbb{E}[X]$.
  2. Compute covariance matrix: $\Sigma = \text{COV}(X)$.
  3. Eigendecompose $\Sigma$: $\Sigma = V \Lambda V^\top$ (where $V$ contains eigenvectors, $\Lambda$ eigenvalues).
  4. Project onto top $r$ eigenvectors: $Y = \bar{X} V_{[:, 1:r]}$.
- **Applications:** Used for visualization and dimensionality reduction.
- **Drawbacks:** Fails to capture non-linear variance.

# t-Stochastic Neighbor Embedding (t-SNE)
- **Overview:** Non-linear dimensionality reduction for 2D/3D visualization.
- **Steps:**
  - Convert pairwise distances into probabilities using Gaussian kernels.
  - Minimize KL divergence between high- and low-dimensional distributions:
    $$ KL(P||Q) = \sum_{i,j \neq i} p_{ij} \log \frac{p_{ij}}{q_{ij}} $$
  - Use t-distribution kernel for robustness to outliers.
- **Hyperparameters:**
  - **Iterations:** Number of gradient descent steps.
  - **Perplexity:** Determines neighborhood size and variance in Gaussian distributions.
- **Drawbacks:** Slower than PCA and sensitive to hyperparameters.

# Multi-Dimensional Scaling (MDS)
- **Overview:** Preserves pairwise distances during dimensionality reduction.
- **Steps:**
  1. Compute pairwise distances: $d_{ij} = ||x_i - x_j||$.
  2. Minimize loss: 
     $$ L(y) = \sum_{i \neq j} (d_{ij} - ||y_i - y_j||)^2 $$

# Clustering
- **Definition:** Group data into clusters where elements in a cluster are more similar to each other than to other clusters.
- **Applications:** Customer segmentation, anomaly detection, image segmentation, etc.

## Metrics for Clustering
- **Within-Cluster Sum of Squares (WSS):**
  $$ WSS = \sum_{j=1}^C \sum_{i \in C_j} d(c_j, x_i)^2 $$
  Measures compactness of clusters.
- **Silhouette Value:**
  Combines intra-cluster cohesion and inter-cluster separation:
  $$ s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))} $$
  Mean silhouette value gives overall clustering performance.

# K-Means Clustering
- **Algorithm:**
  1. Initialize $k$ cluster centroids randomly.
  2. Assign points to the nearest centroid.
  3. Update centroids as the mean of assigned points:
     $$ c_k = \frac{1}{n} \sum_{i \in C_k} x_i $$
  4. Repeat until centroids stabilize.
- **Assumptions and Failure Cases:**
  - Assumes spherical clusters with equal variance.
  - Struggles with anisotropic or unevenly sized clusters.
  - Sensitive to initial $k$ and fails with complex shapes.

# DBSCAN Clustering

## Introduction
- **Limitations of K-Means**: Struggles with non-convex or non-linear clusters as it relies solely on distance from centroids.
- **DBSCAN Strengths**:
  - Clusters based on **density**.
  - Handles arbitrary shapes, which K-Means cannot.
- **Key Feature**: Can identify noise and outliers.
- **Full Form**: Density-Based Spatial Clustering of Applications with Noise.

## Basic Concepts

### Hyperparameters
- **ε**: Distance threshold defining point connectivity.
- **minPts**: Minimum number of points required to form a **core point**.

### Core Point
- A point **p** is a core point if there are at least `minPts` within a distance of `ε`.

### Direct Reachability
- Point **q** is **directly reachable** from **p** if:
  - **q** is within **ε** of **p**.
  - **p** must be a core point.

### Reachability
- A point **q** is reachable from **p** if there exists a chain of points connecting them.
- Reachability is **not symmetric**.

### Outliers and Noise
- Points that are:
  - Not core points.
  - Not reachable from any core point.

### Density Connection
- Points **p** and **q** are density-connected if:
  - Both are reachable from a common core point.

### Cluster Definition
- A cluster in DBSCAN satisfies:
  1. All points in the cluster are **density-connected**.
  2. Every point in the cluster is **reachable** from a core point.

![Basic Connectivity Example](Insert image showing connectivity in DBSCAN.)
# DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

## Introduction
- K-Means clustering fails for non-convex or non-linear clusters as it relies on centroids and distance measures.
- DBSCAN clusters based on **density**, allowing for arbitrary cluster shapes.
- DBSCAN = Density-Based Spatial Clustering of Applications with Noise.

---

## Basic Concepts

### Hyperparameters
- **ε (Epsilon):** Distance threshold to define connectedness between points.
- **minPts:** Minimum number of points to define a core point.

### Core Point
- A point `p` is a core point if at least `minPts` are within distance `ε` of it (including `p`).

### Direct Reachability
- A point `q` is directly reachable from `p` if `q` is within distance `ε` and `p` is a core point.

### Reachability
- A point `p` is reachable from another point `p` if a chain of direct reachability exists between them. **Note:** Reachability is not symmetric.

### Outliers and Noise
- Points not reachable from any core point are classified as **noise**.

### Density Connection
- Points `p` and `q` are density-connected if both are reachable from a core point `o`.

### Cluster
- A cluster in DBSCAN:
    - All points are mutually density-connected.
    - All points are density-connected to at least one point in the cluster.

![Example Non-Linear Clusters](Insert image showing DBSCAN clustering of non-linear shapes.)

---

## Basic Algorithm

1. Identify all core points based on `ε` and `minPts`.
2. Form connected components of core points.
3. Assign each non-core point to the nearest cluster if within `ε`; otherwise, classify it as noise.

![Example Non-Linear Clusters](Insert image showing DBSCAN clusters with noise points.)

---

## DBSCAN Hyper-parameters

### Distance Threshold (ε)
- Defines connectedness. 
- Use the **k-distance plot** to determine an optimal `ε`.

### Minimum Number of Points (minPts)
- Minimum `minPts ≥ d + 1` (where `d` is the input dimensionality).
- Larger `minPts` result in more stable clusters. Common heuristic: `minPts = 2d`.

![K-Distance Plot](Insert image showing sorted k-distance plot.)

---

## Advantages
1. No need to pre-specify the number of clusters.
2. Can find clusters of arbitrary shapes.
3. Handles noise and outliers effectively.
4. Efficiently accelerated with spatial data structures (e.g., KD-Trees).

## Disadvantages
1. Two hyperparameters (`ε`, `minPts`) require tuning.
2. Relies on assumptions about data density.
3. Explicit outlier definitions may misclassify data.

![[Pasted image 20250109161224.png]]
### What You Must Learn

- **Dimensionality Reduction**: 
    - What it is and why to use it.
    - Key Methods:
        - PCA: Know well.
        - t-SNE, MDS: Understand basics and intuitions.
    - Comparison: PCA vs. t-SNE vs. MDS?

- **Clustering**: 
    - What it is and why to use it.
    - Key Methods:
        - K-Means and DBSCAN.
    - Understand (Dis)advantages of the two methods.

---

### Questions to Think About

1. What is the difference between PCA and t-SNE?
2. How to determine the target dimension for PCA?
3. How do you decide how many clusters are optimal for your application?
4. On what type of data distribution does DBSCAN outperform K-Means?
5. Does changing the distance function in a clustering method change the clusters (assuming the same number of clusters)?

---

### Take Home Messages

- **Dimensionality Trade-offs**:
    - Increasing the dimensions does not always lead to higher performance!

- **Challenges in Unsupervised Learning**:
    - It is **hard**.
    - Requires manual tuning of hyperparameters and selection of methods.
    - Trial and error is often necessary:
        - Example: Experiment with different distance metrics when clustering and inspect the results.
