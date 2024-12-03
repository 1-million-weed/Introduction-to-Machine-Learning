---
tags:
  - Marinus
  - Lectures
  - Matthijs
  - _FirstPass
---


{karolina yaps}


# Linear regression
---
## Representing the dataset
- The dataset is made up from $N$ pairs of **observations** and **responses** (inputs and outputs)
- The response is a row vector of $N$ scalars 
- We can group together the observations throughout the dataset in a matrix $X \in \R^{n X d}$ #_Err_Design 
	- Matrix are referred to with capital letters and vectors with lowercase
- The basic linear regression model is $f(x)=\hat{y} = ...$ #_Err_Design 
- The total number of parameters/weights is $P = d + 1$, where d = dimensions of input features
  *a.k.a: parameters = features + 1*
>[!info] The idea is to set the weights to make the line of best fit fit the data

### Matrix formulation
- Input features $x^~ = (x_1, ..., x_p) = (x ,1)$
- Weight vector $w=$$\begin{pmatrix} w \\ b \end{pmatrix}$
Once above is done:
- $f(x)=x\~w\~ = \vector{x 1}$ #_Err_Design 

## Closed form solution
*By condensing $x$ and $y$ for the whole dataset (incl. 1 col) we can get: $L=(y-Xw)^T(y-Xw)$*
- Linear regression models are trained using [[MSE]] loss 
- Start with loss function to find value W (that minimizes loss)
  Derivative(L) = 0 Gives $W=n^{-1}(X^TX)^{-1}X^Ty$
	- Here $X$ is a n x (d+1) matrix
	- Rows of $X$ have to be **linearly independent** (data points are not products of each other)
- Linear independence is easily broken -  so we need to constraint the data points (independency)


## Modelling assumptions 
- ### Residuals
	- The actual model considers errors / residuals $\epsilon$ : $f(x_i)=x_iw+\epsilon_i=y_i$
	  if $\epsilon$ is 0, the model is a perfect fit
- ### Data must not have dependencies
	- The rows of $X$ have to be linearly independent
- ### Residuals have to be independent
	- The residuals $\epsilon_i$ are assumed to be **independent** (also from input) 
- ### Linearity
	- Input features $x_i$ are treated as fixed values.
	- Input features can (because they are treated as fixed values) be transformed to produce other features
		- Thus we can work around linearity
- ### Constant Variance
	- [[Variance]] should stay the same, and not dependent on inputs or outputs of the model
		- This is called homoscedasticity
	- Variance of the output or the errors can vary with the input or output of the model
		- This is called heteroscedasticity
		- For example: if larger outputs have larger variance then smaller outputs

- Different data sets / training sets can have the same regression line. 
	- A linear regression model should be used with care
## Solutions with Gradient Descent
- Used as an alternative to an intractable analytic solution
- Set the loss to 0
- Compute the gradient of the loss with respect to the parameters
	- Closed form: $\delta L/\delta w=n^{-1}X^T(Xw-y)$
- We slowly step the parameters closer with:
- $w_{m+1}=w_m-\alpha n^{-1}X^T(Xw-y)$
	- Initialize $w$ with initial random vector (in small range)
	- $\alpha$ is the learning rate (tuned manually)
	- $m$, $m+1$ are iteration indices
- iterations to get as close to 0 (gradient)
- gradient points in direction of maximum increase (visa versa for negative side)
- does not ensure global optimisation
> chatGPT uses this framework
- hyperparamer - no of iterations
![[Pasted image 20241125193618.png]]
The $x_i$ is the steps we take to hone in on the parameter
## Multivariable Linear Regression
- If labels $y$ are not scalars, but vectors of size $m$X1
- Then: For every output variable, you do one regression line. 
	- $f(x)=[x_k*w_k+b_k]^K_{k=1} = XW+b$
	  
	- So, if you have 3 variables, you should make three different regression models
	- One neuron in a neural network is basically one linear regression model running

## Interpretability of Weights
- For linear models like LR, the weights can be interpretable if:
	- They are normalised to be in same range 
	- The output features should also be normalised/scaled to be at same range
- A weight close to 0 is basically not important
- If you can normalise the features, you can interpret it
	- this is called generalising the model 

### Polynomial regression
- Polynomial regression: 
	  A polynomial of degree p:
	  $f(x)=\sum^p_{i=0}W_ix=w_0+w_1x+w_2x^2+...+w_pX^p$
	- linear model on features $[1, x^1, x^2, x^3,...,x^p]$
		- Feature space dimensionality increases by a factor $p$

## Outliers in Linear regression
- One outlier can have a big impact on the weights
- robust methods & alternatives against outliers: 
	- student's t-distribution or heavy tailed distribution instead of Gaussian distribution
	- Random Sampling Consensus (RanSaC) is used to fit LR models, and detect outliers
	- Exploratory data analysis can be used to identify and remove outliers 

# Logistic regression ([[Classification]] model)
Probabilistic model 
- Decision boundary, shape of boundary where 
- if data are seperatable possible infifinate decision boundary 
- extension of linear regression for classification 
- #_Err_Mismatch insert sigmoid function

## Probabilistic interpretation
outputs continious value in range $[0,1]$
- We dont do conditional probs 

## Training logistic regression
- uses binary cross-entrapy loss
- logit - inputs to the logistic/sigmoid functions
	- #_Err_Mismatch insert formula 23 and 24

## Training the gradient descent
- formula 25
- 26
- initialise initial weights
- very similar to linear regression except for log function
- learning rate and number of iterations are important
- loss will never reach zero

## Multi-class logistic regression
- ove vs one 
	- each pair of classes train one classifier 
	- decide output, select class with most votes
	- requires... many...
- one vs all
	- ...
- one vs rest(all) 
	- better for more classes

## Multinomial logistic regression
- softmax function 
- simpler way to frmulate multi-class regression
- some complicated and too fast shit... im not learning much from here
- output sums to 1.0

## Training multnomial LR
- categorical cross-entropy
	- one-hot encoded
	- predictions all sum to one
- gradient has the same form as bianry ml

## Multilabel logistic classification
- multiple classes at the same time 
- independent properties
- binary vector - not one hot encoded
- classes are only decided to be present by setting a threshold on the probability and if its over the class in present else not.

# Support Vector Machines (SVMs)
- noone uses them anymore
- useful in loading big data points? didnt get that
- SVMs are robust to outliers (can maximise separation distance)
- Models decision bound 
- kernel trick - non linear models

## Maximum Margin Formulation
- constains model to introduce one margin that is unique
- look at lecture slides. good explanation about mirgin width.
	- width is 2 devided by abs of weights
- xw+b=1 positive class
- xw+b=-1 negative class
- in order to max the margin we need to min the ||w||
- use l2 norm
- equation 35 to 38
- to predict we simply look at the sign of prediction. positive or negative
- if the points are not linearly seperatable this is not possible 

## Soft Maximum Margin Formulation
- can be used for non-linearly seprable data
- hinge loss
- regularizatioin labels
- constraint optimisation (difficult, a lot of math)
- the smaller the c or k the larger the margin width
- looks like a c between 0.00001 and 0.1 are optimal

## SVM Concepts

>margin 

> support vectors

## Training SVMs

>[!def] Hard Margin 
> Quadratic programming margin

>[!def] Soft Margin

## Multiclass SVMs

one vs rest and one vs one

## Support vector Regression 
- dont dive deep into this 

# Linear Discriminant Analysis (LDA)
- Bayesian approach
- models distribution for each class separately. previous all classes were modelled together
- distance between means 

## Estimating the Gaussian Mean
- mean of each class class 
- covariance matrices for each class
- assumption of homoskeasticity
- data needs to be normalised

## LDA Concept

## LDA Class Separation

## Fisher's Criteria or Discriminant
## LDA Model/Equations
## Questions to think about
- similar to exam questions
- 