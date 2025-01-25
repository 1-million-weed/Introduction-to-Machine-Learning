---
tags:
  - Marinus
  - _FirstPass
  - Lectures
  - LLM
---
Got here and he asked me what a feature is 
There are 16 ppl here

# Questions - Intro to ML

## Questions and Answers

1. **What is a feature?**
    - A feature is an individual measurable property or characteristic of the data being used for modeling. Features are the input variables used by machine learning models to make predictions.

2. **Describe linear separability in your own words.**
    - Linear separability refers to the ability to separate data points of different classes using a straight line (in 2D), a plane (in 3D), or a hyperplane (in higher dimensions). It means that a linear classifier, like logistic regression or a perceptron, can perfectly distinguish between the classes.

3. **What is the need for normalization and pre-processing of features?**
    - **Normalization:**
        - Scales features to a similar range, which is crucial for algorithms that depend on distance metrics (e.g., k-NN, SVM).
    - **Pre-processing:**
        - Improves data quality, handles missing values, removes outliers, and ensures that features are suitable for model training.
        - Helps models converge faster and perform better.

4. **Can regression labels/targets be normalized?**
    - Yes, normalizing regression labels/targets is common when targets have a wide range of values, especially if the model assumes numerical stability. Normalization ensures consistent scaling and helps the model learn more effectively.

5. **What factors can hurt the performance of a Machine Learning model?**
    - Poor data quality (e.g., noisy, missing, or irrelevant features).
    - Overfitting or underfitting due to inappropriate model complexity.
    - Inadequate feature engineering or feature selection.
    - Imbalanced data, leading to biased predictions.
    - Incorrect hyperparameter tuning.
    - Lack of sufficient or diverse training data.

6. **Your model does not learn, what is the most likely cause?**
    - Common causes include:
        - Learning rate being too low or too high.
        - Data issues (e.g., incorrect labels, insufficient data).
        - Model architecture being too simple to capture data patterns.
        - Initialization problems (e.g., poor weight initialization).
        - Features not properly pre-processed or scaled.
 
# Supervised Learning I - Linear Models

## Questions and Answers

1. **What is the basic concept underpinning SVMs?**
    - Support Vector Machines (SVMs) aim to find the optimal hyperplane that maximizes the margin between data points of different classes. The support vectors are the data points closest to the hyperplane.

2. **How do ML Classification methods relate to Linear Separability?**
    - Linear classifiers, like logistic regression or SVMs, can only separate data that is linearly separable. For non-linear data, methods such as kernel-based SVMs or non-linear classifiers are required.

3. **Explain the concept of the Kernel Trick and its relationship with Kernel Functions.**
    - The kernel trick allows SVMs to map input data into a higher-dimensional space without explicitly computing the transformation. Kernel functions compute the dot product in the transformed space efficiently, enabling SVMs to handle non-linear data.

4. **How to transform a binary classifier into a multi-class one?**
    - Use strategies like:
        - **One-vs-All (OvA):** Train one classifier per class, treating it as a binary classification problem against all other classes.
        - **One-vs-One (OvO):** Train one classifier for each pair of classes and use majority voting for classification.

5. **What are the main assumptions behind LDA?**
    - Linear Discriminant Analysis assumes:
        - Data is normally distributed within each class.
        - Classes have the same covariance matrix.
        - Features are linearly separable with respect to their class means.

6. **How is it possible to interpret the coefficients of linear regression?**
    - Coefficients indicate the change in the target variable for a one-unit increase in the corresponding feature, holding all other features constant.

7. **What is the difference in the equations of linear and logistic regression and what are its effects on the characteristics of the two models?**
    - **Linear Regression:**
        - Predicts continuous values using a linear equation: $y = \beta_0 + \beta_1 x_1 + ... + \beta_n x_n$.
    - **Logistic Regression:**
        - Models probabilities using a sigmoid function: $P(y=1) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + ... + \beta_n x_n)}}$.
        - Outputs probabilities and is suitable for classification tasks.

---

# Supervised Learning II - Non-Linear Models

## Questions and Answers

1. **Why is k-NN non-linear?**
    - The shape of the decision boundary in k-NN is determined by the data distribution. Since k-NN classifies based on proximity to neighbors, its boundaries are often irregular and non-linear.

2. **Why does an ensemble improve performance?**
    - Ensembles combine predictions from multiple models, reducing the likelihood of errors. By averaging or voting across models, ensembles mitigate individual model mistakes and increase robustness.

3. **Why do Random Forests often outperform Decision Trees?**
    - Random Forests, being ensembles of decision trees, are less prone to overfitting. Each tree is trained on a random subset of the data, and predictions are averaged, reducing variance and improving generalization.

4. **Are Decision Trees linear or non-linear methods?**
    - Decision Trees are non-linear methods because they split data iteratively based on feature thresholds, creating complex and non-linear decision boundaries.

5. **How does the decision boundary of a Random Forest look like, considering the decision boundary of a Decision Tree?**
    - A single Decision Tree creates jagged, piecewise-linear decision boundaries. Random Forests aggregate these boundaries from multiple trees, resulting in smoother and more robust non-linear boundaries.


# Evaluation and Cross Validation

1. What are the differences between cross-validation, k-fold cross-validation, and leave-one-out cross-validation?
	1. Not sure, but look at the slides
	1. k-fold - 1 dataset k eaqually sized folds and then train on 
	2. blind training data, train all k -1 folds then 
2. What is the major difference between a validation and test set?
	1. tune hyperparameters and test your model
1. Give an example (not ones from the lecture) of leakage between train and test set.
	1. Something about leaving a human out of a train/test split. for instance blood sample tests to match a person to their blood. 
	2. do the splitting first then apply the pre processing steps
2. What is overfitting?
	1. bruh
3. How can leakage be detected?
	1. there is no clear way to detect it, just handle your data right
4. What should you consider when selecting cross-validation methods for your problem?
	1. depends on your data/dataset size

# Statistical Learning Theory

1. What is overfitting in your own words?
2. How is overfitting different from underfitting?
	1. model underrepresents the concepts we are trying to catch 
	2. underfitting you dont fir the training set 
	3. overfitting, you arent fitting the validation set
3. What variables influence the learning problem success? And how?
	1. Dimensionality of the features
	2. Number of parameters
	3. influence? P > Dpts.
	4. Everything has to be less than the number of training points
4. What is the VC Dimension?
	1. lost
5. Your model overfits, suggest some steps to improve.
	1. less parameters
	2. change training points, dimensionality, features
6. Conceptual question: We would show you some train/val loss curves and ask what over/underfitting is happening.
	1. cool

# Regularization

1. What is the intuition behind regularization?
2. What is the difference between parameter and output regularization?
	1. constraints on the parameters
	2. constraints on the output. implicit puts constrains on the parameters
	3. where the constraint is put 
3. What is the relation between regularization, model complexity, and generalization?
	1. constraints on parameters: less values: less complex: helps with overfitting: helps with generalization.
4. How to select the regularization strength coefficient $R / C / \lambda$?
	1. via cross validation
	2. balance between over and underfit
	3. use cross validation to find the one with the best performance
5. How does decreasing/increasing the regularization strength $R$ affect under/overfitting?
	1. inversely proportional to amount of parameters
6. How to apply regularization to a multi-class classification model (say multi-class LogReg)?
	1. penalty on the weights? or something
7. Why does multi-task learning improve performance?
	1. Uses more knowledge 
	2. fitting multiple tasks
	3. more regularisation 
	4. more constraints
	5. more labels


# Neural Networks

## Questions and Answers

1. **What are the differences between MLPs, CNNs, and RNNs?**
    - **MLPs (Multi-Layer Perceptrons):**
        - Fully connected networks.
        - Suitable for tabular data and tasks without spatial/temporal structure.
    - **CNNs (Convolutional Neural Networks):**
        - Designed to process spatially structured data like images.
        - Utilize convolutional layers to capture spatial hierarchies.
    - **RNNs (Recurrent Neural Networks):**
        - Designed for sequential data like time series and text.
        - Use recurrent connections to capture temporal dependencies.

2. **Why do neural networks suffer from vanishing gradients?**
    - The issue arises when gradients of the loss function diminish as they are propagated back through layers, especially in deep networks with activation functions like sigmoid or tanh.
    - This leads to minimal weight updates in early layers, stalling training.
    - Solutions include:
        - Using activation functions like ReLU.
        - Employing normalization techniques (e.g., Batch Normalization).
        - Initializing weights properly.
        - Using architectures like LSTMs or GRUs for sequence models.

3. **You train a neural network in some regression dataset, and the training loss stays constant. What can be the issue and how do you solve it?**
    - **Potential Issues:**
        - Learning rate is too low or too high.
        - Model is underfitting due to insufficient complexity.
        - Data preprocessing errors (e.g., incorrect scaling or normalization).
        - Weights are stuck in poor initialization.
    - **Solutions:**
        - Tune the learning rate.
        - Check and preprocess the dataset.
        - Increase model complexity (add more layers or neurons).
        - Experiment with better weight initialization strategies.

4. **What are Long-Term Dependency problems and in which kind of NN do they happen?**
    - **Problem:** The inability of RNNs to effectively learn and retain information over long sequences due to vanishing gradients.
    - **Occurrence:** Common in standard RNNs when sequences exceed a certain length.
    - **Solution:** Use LSTM or GRU architectures which have gating mechanisms to preserve information over longer sequences.

5. **What are the differences between a standard RNN and a LSTM?**
    - **RNN:**
        - Simple recurrent architecture.
        - Suffers from vanishing gradients.
    - **LSTM (Long Short-Term Memory):**
        - Enhanced architecture with **gates** (input, forget, and output) to control the flow of information.
        - here the big point is the gates!
        - Better suited for learning long-term dependencies.
        - Overcomes vanishing gradient issues.

6. **What kind of neural network would you use to perform classification of videos?**
    - Use a combination of architectures:
        - **CNNs** to extract spatial features from video frames.
        - **RNNs (e.g., LSTMs/GRUs)** to model temporal dependencies across frames.
        - Alternatively, 3D CNNs or Transformers (e.g., Vision Transformers or TimeSformer) can be used to process spatiotemporal data directly.


# Deep Learning

## Questions and Answers

1. **Give a definition in your own words of Deep Learning.**
    - Deep Learning is a subset of machine learning that uses neural networks with multiple layers to model complex patterns in large datasets. It is particularly effective for tasks such as image recognition, natural language processing, and time-series prediction.
    - non-linear layers 
    - feature hierarchy that is learnable

2. **Give three difficulties of training deep networks.**
    - **Vanishing or Exploding Gradients:**
        - In deep networks, gradients can become very small or large during backpropagation, making training unstable.
    - **Overfitting:**
        - Deep networks can memorize the training data instead of generalizing to unseen data.
    - **High Computational Cost:**
        - Training deep networks often requires significant computational resources and time due to the large number of parameters.
    - **Optimisation problems**:
	    - local minima 

3. **How does dropout work and what is its effect?**
    - **How it works:**
        - Dropout randomly disables a fraction of neurons during each forward and backward pass of training.
    - **Effect:**
	    - Regularization effect
        - It reduces overfitting by preventing co-adaptation of neurons, encouraging the network to learn more robust features.

4. **What is the importance of residual connections in an architecture?**
	- Skip connection or something
    - Residual connections help by:
        - Allowing gradients to flow directly through the network, mitigating vanishing gradient issues.
        - Enabling the training of very deep networks by making it easier to optimize.
        - Promoting feature reuse across layers, improving performance and convergence.

6. **What is an encoder-decoder architecture and for what kind of data can it be used?**
	- Sequence in and out. different length. encodes in hidden state vector, then to decoder, and then different length output.
    - **Definition:**
        - An encoder-decoder architecture consists of two components: 
            - The encoder compresses the input data into a compact representation.
            - The decoder reconstructs the target output from this representation.
    - **Applications:**
        - Sequence-to-sequence tasks like machine translation, text summarization, and image captioning.

8. **What is attention? What is the concept of an optimizer?**
    - **Attention:**
        - A mechanism that allows the model to focus on specific parts of the input data when making predictions. It is widely used in tasks like NLP and computer vision.
    - **Optimizer:**
        - An algorithm used to update the model’s weights during training by minimizing the loss function (e.g., SGD, Adam, RMSProp).

9. **What are the differences between soft-attention and multi-head attention?**
    - **Soft-Attention:**
        - Computes a weighted sum of all input features, focusing on the most relevant parts.
    - **Multi-Head Attention:**
        - Extends soft-attention by using multiple attention mechanisms in parallel. Each head learns to focus on different parts of the input, capturing richer patterns.

# Unsupervised Learning
little note: i feel bad cause the woman is reading out the questions expecting us to answer but i really did not care during this fucking lecture. I dont like this woman.
## Questions and Answers

1. **What is the difference between PCA and t-SNE?**
    - **PCA (Principal Component Analysis):**
        - Linear dimensionality reduction technique.
        - Captures variance in the data by projecting it onto orthogonal components.
        - Preserves global structure.
    - **t-SNE (t-Distributed Stochastic Neighbor Embedding):**
        - Non-linear dimensionality reduction.
        - Focuses on preserving local neighborhoods in the data.
        - Suitable for visualizing high-dimensional data in 2D/3D.
        - deriving probabilities

2. **How to determine the target dimension for PCA?**
    - Choose the number of principal components such that the cumulative explained variance(magnitude eigen values/ distance eigen vectors) ratio exceeds a certain threshold (e.g., 90-95%).
    - Use a scree plot to identify the "elbow point" where the explained variance gain starts diminishing.
    - Eigen values

3. **How do you decide how many clusters are optimal for your application?**
    - Use methods like:
        - **Elbow Method:** Plot the within-cluster sum of squares (WCSS) for different values of $k$ and find the elbow point.
        - **Silhouette Score:** Measure how similar samples are within their cluster compared to other clusters.
        - **Gap Statistic:** Compare the total within intra-cluster variance to a null reference distribution.

4. **On what type of data distribution is DBSCAN outperforming k-means?**
    - DBSCAN is more effective on:
        - Data with irregularly shaped clusters.
        - Data with varying density.
        - Data containing noise and outliers, which DBSCAN can handle by labeling them as noise.

5. **Does changing the distance function in a clustering method change the clusters? (Assuming the same number of clusters).**
    - Yes, the choice of distance function (e.g., Euclidean, Manhattan, Cosine) can significantly impact the clustering outcome as
