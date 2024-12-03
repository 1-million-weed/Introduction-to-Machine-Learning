---
tags:
  - Marinus
  - _FirstPass
  - Lectures
---

# Kernels
- kernel can transform the SVM from a linear classifier into a non-linear model. 
	- done by changing the shape of the decision boundary
- Kernel trick is a mathematical trick
- a lot of math and a lot of math yap
- a simple function that represents a high dimension dot product
- kernel function should be a function of another function applied to a dot product
- $$f(x)=x\cdot w + b$$
- becomes:  $$f(x)=\phi (w) \cdot \phi (x) + b$$
- Then we can sub in the previous equation $$f(x)=(\sum^{n}_{i=1}x_iw_i+c)^2 $$
- form his example i understand that it takes some vectors with points and transforms them to be able to linearly separate them:
![[Pasted image 20241121151546.png]]

- Matthijs asked a smart question. 
	- "are the kernal functions the same for all model?"
	- "No, different for every model, lets move on"
- process is called kernalisation 
- metric how well the data is optimised to find the kernal function 
- trial and error
- Kernal function is defined as: $$K(x,y)=\phi(x)\cdot\phi(y)$$
- Works because data is projected/transformed into a higher dimensional space
- Kernal functions examples:
	- Polynomial Kernel: $$K(x,y)=(x\cdot y+c)^d$$
	- Gaussian Kernel or Radial Basis Function (RBF): $$k(x,y)=exp(-\frac{(||x-y||)^2}{2o^2})$$
	- Laplace RBF: $$K(x,y)=\exp(-\frac{||x-y||}{2o^2})$$
	- Hyperbolic Kernel: $$K(x,y)=\tanh(\alpha x\cdot y+c)$$
## Parameters of RBG Kernel
![[Pasted image 20241121153115.png]]

# Nearest Neighbour Methods

- talk about distances
- quick look... lets go
- Euclidean distance
	- l2 norm 
	- ruler distance
- Manhattan Distance
	- from how the cities are built in the US
	- 90 degree streets
	- l1 norm
	- $$=\sum^{}_{}$$
![[Pasted image 20241121155118.png]]

- K-Nearest Neighbours (supervised)
	- lazy method
	- takes very long - has to go through every data points
	- data driven 
	- k is the only parameter
	- in which class does a datapoint fit
		- measure distances (pick distance measure) for all points (hence long compute time)
		- allocate the new points to the class with the smallest distance
	- How do we determine the most optimal value for K?
		- dependent on the amount of classes
		- 2 classes, k has to be odd
	- we need to normalise the data. normalisation scaling
	- We know our labels here

# Ensemble methods

## Improving machine learning models

- overfits? how do we fix this? what are our options?
- more perspectives
- ensamble models do this. Combing different models (getting different expert opinions)
	- expectation that multiple models predictions should be better than single model

- Homogeneous Ensambles
	- similar in structure
	- parameters are different
	- copies of same models

- Heterogeneous Ensambles
	- Different in nature
	- do not share structure or model equations

## Basic ensamble methods

- Voting model
	- copies of same models
	- different params
	- takes average prediction between models
- Boositing
	- weak learners 
	- simple ML models
	- combined in sequence 
	- weak models learn to correctly predict what previous models incorrectly predicted 

## Decision Trees

- Know difference between classification/regression, continuous/categorical
- Genie index
	- entropy, how pure is the outcome
	- how similar are the things you get in that node
- Tree like model, illustrates series of events that lead to certain decisions.
- recursively repeat this step until you secure decision
- important to identify attributes that have the most meaningful outcome for the problem.
- Pros
	- small easy to interpret
	- scale well to large N 
	- can all types of data
	- automatic variable selection
	- can handle missing data
	- completely non parametric
- cons
	- large treees can be difficult to interpret 
	- all splits depend on previous split
		- capturing interactions :D - additive models :)
	- are step functions
	- single trees have poor predictive acuracy 
	- single trees have high varience

## Random Forrest 

- Bagging (Bootstrap Aggregating)
	- random subsets of data to create N smaller data sets
	- fit decision tree on each subset
- Random Subspace Methods (Feature Bagging)
	- Fit N different decision trees by constraining each one to operate on a random subset of features
![[Pasted image 20241121161331.png]]

- pros
	- Competitive performance
	- Remarkably good ”out-of-the-box” (very little tuning required)
	- Built-in validation set (don’t need to sacrifice data for extra validation) 
	- Typically does not overfit 
	- Robust to outliers 
	- Handles missing data (imputation not required) 
	- Provides automatic feature selection 
	- Minimal preprocessing required
- cons
	- • Although accurate, often cannot compete with the accuracy of advanced boosting algorithms
	- Can become slow on large data sets
	- Less interpretable (although this is easily addressed with various tools such as variance importance, partial dependence plots, LIME, etc.)

**Questions to think about**
1. Why knn non linear?
	1. does not have a linear decision line
2. Why ensamble improve?
	1. multiple models, multiple predictions, averaged answer of predictions. better answer
3. Random forest outperform decision tree?
	1. more trees. better answers. Reduces overfitting
4. Decision trees linear or non-linear?
	1. non-linear. 
5. Difference of decision tree boundary of decision tree and random forest
	1. Decision boundary becomes more non-linear

Some of these 100% going to be in the exam!!

**Take Home**
- why and how kernel
	- to transform non linear to linear SVM
	- dot product of some vectors in some dimension in space
- so just about every decision depends on the data. and everything is high dimensional so you cannot visualise it to decide so youre just guessing
