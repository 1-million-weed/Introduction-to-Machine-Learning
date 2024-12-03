---
tags:
  - _FirstPass
  - Lectures
  - Marinus
---
# Training parameters
- Training samples - size of training set (in total)
- Feature dimentionality - how many features do you have
- Model parameter - how many elements in weights and bias
- n less than d, you wont have enough variation in the features (N and D refer to the previous three points)
- the more data points the better. at least D
- N needs to be much bigger than D
- How about P? Pushing P hmmm
- you should have at least n data points with regards to P
- !! Dont forget about these. apparently they are important
- P much much larger than N
	- more p than you need 
	- oof

# Rules of Thumb

- N~10P
- N~2D 
no theory, practical knowledge

# Imbalanced class data

- you need more data of a specific class to balance the data
- use weights in the loss funciton. thety counteract the effect of each classified value
- set them as the inverse for the class with less samples
- losses are averages of the dataset
- Weighted averages. Used by deep learning frameworks
- it works if the imbalance is not too large like 8:2 is fine
- logg if the weights are umbalanced
- sampling balancing 
	- sample weights
	- guestimate
	- inverse of frequency 

## Smote
oversampling
- new synthetic samples of minority class
- use knn
- random factor in [0,1]
- repeat until dataset is balanced
- not in hightly diemnsional classes
- scikitlearn

## Data augmentation
- apply for minority class 
- can only be applied for training due to #leakage 

# How many samples?

>fundamental machine learning thworym is that you cannot train a model without making any assumptions
- Hard mins
- **More training data increases generalization and overall model performance**
	- We need MOOOORRRREEEEEE
- by doing transfer learning you can have a 10% performance increase with only sample per class
	- refer to lecture notes and transfer learning
- Factors that effect classification difficulty
	- how different are the samples (bird species or birds and cats)
	- more on lecture slides
	- different models have different training curves
- Interclass
	- variation between different samples (animals)
- intraclass
	- variation within the same classes  (owl species)

# Effect of feature dimensionality 

> Cover's theorym

- computes prob of n data points of d dimensionod what is the probability of the datapoitns being linearly independent
- looks like a bunch of sigmoid curves
- more features are better but then you need moer data poitns
- kernel trick can help with this. 
	- this is the motivation or the kernel trick
- more data more features
	- liinearly seperapable
- fundamental

# Overfitting

- part of evaluation is detecting overfitting and then dealing with it from there
- memorised training data or noise in dataset
- reduces generalisation
- its a continuous phenomena
- learn how to detect it properly

## Cross validation

- training and test set

## Generalization
- model works outside of the training set then it generalizes

## Overfitting

- always a gap between train and validation/test losses
- lgap=lval-ltrain #_Err_Mismatch 
- no way to completely avoid overfitting

## Causes

- little data
- misspecification
- incorrect features
- model too large for data
- no gaurantee

## Solution 
- missed this

## Overfitting

![[Pasted image 20241128153519.png]]
2, 3 and 6 parameters. 
3 wins

## The golden Rule

- validation much larger than training set, overfitting
- slightly larger is normal
- look at the gap
- we need to find the point where we have the most generalisable model that is not overfitting. differential calculus. find f'=0![[Pasted image 20241128153809.png]]
- so just about everyhting depends on the data 

# What is not overfitting

- i clocked out. getting matthijs a present
# No free lunch theorym (NFL)

- in the average, no leaning algorithm is superior to another

## Bias variance trade-off

very big very long equation

something about noise 
if one label is incorrect this does not work for example
if the model assumptions aren't enough, the model underfits
if the bias is low, high variance, overfitting
need to find the trade off for the minimal error
middle ground
can never reach zero because of some error 

> Bias: get

> Varience: get

small variance has large bais, large varience has small bias

## Double descent 

- in some cases the evaluation loss goes up then back down
- ![[Pasted image 20241128154915.png]]
- happens with 

# Generalization Theory

## Shattering
- if it perfectly classiefs the dataset it shatters it 
- shutters 3 points but not 4
- VC-dimentionality
- different models have difference VC dimensions
- multilayer perceptrons 
- why?
	- one way to measure classifier complexity
	- more for linear models
	- "kind of interesting"
- Theoretical bounds
	- i am not understanding this
	- test error is larger than the training error
	- more data points the model should improve
- what's the point?
	- how it is defined
	- relationship between variables
	- not useful in practive (then why do we use it)
	- mdoel selection is important, each has different bounds

# what we should learn form this lecture
- relationship between N, D, P
- Cover's theorem
- concept of overfitting, most important in machine learning!!
- bias-variance tradeoff and its relationship to over/underfitting
	- at least 2 questions about overfitting
- !!we will have to draw curves that show over and underfitting #exam
