---
tags:
  - Marinus
---

karolina yaps

# Linear regression

- dataset from N pairs
- row vectors 
- group rows to form BIG X matrix. 
- Matrix are capitals and vectors are lowercase
- number of weights is d + 1
- idea is to set the weights to make the line of best fit fit the data
## closed form solution

- lin reg uses MSE loss 
- start with loss frunction 
- transform into matrix product
- Add math here (7, 8, 9, 10)$$L=\frac{}{}$$ #_FirstPass 
- need to remember to check the dimensions
- rows are linearly independent (data points are not products of each other)
- put constraints on the datapoints
- `i have smooth brain`

## Modelling assumptions 
### Residuals
- 
### Data must not have dependencies
### Residuals have to be independent
### linearity
### Constant Varience
- called Homoscedasticity

## Solutions with gradient descent
- compute the loss
- set it to 0
- start with initial weights
- iterations to get as close to 0 (gradient)
- gradient points in direction of maximum increase (visa versa for negative side)
- does not ensure global optimisation
- chatGPT uses this framework
- hyperparamer - no of iterations

## Multivariable Linear Regression
- for every output variable you do one regression line. one for every output.
	- so if you have 3 variables that will be three different regression models
- one neuron in a neural network is basically one linear regression model running

## Interpretability of weights
- interpretable
	- normalise (i told you to do it) to be in same range 
	- output features should also be normalised/scaled to be at same range
- when a weight is 0 it is basically not important
- if you can normalise the features, you can interpret it
	- this is called generalising the model 

### Polynomial regression
- poly of degree p
- sum of weights and points sqaured(p) up to the degree p
- linear model on features $[1, x^1, x^2, x^3,...,x^p]$

## Outliers in Linear regression
- one outlier can change the weights
- are robust methods against weights 
	- t-distribution or heavy tailed distribution instead of Gaussian distribution
	- Random Sampling Consensus (RanSaC)
	- Exploratory 

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