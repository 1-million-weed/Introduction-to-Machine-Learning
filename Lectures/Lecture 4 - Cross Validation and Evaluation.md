---
tags:
  - _FirstPass
  - Lectures
  - Marinus
---
# Cross Validation Methods

## Learning Performance
>[!def] Losses
>Function that guides learning during optimization process. Defines task to be learnt and quality of solutions. usually differentiable.

>[!def] Metrics
>Measurements of quality that let the ML developer and users of the model evaluate the the success of the learning. May be non-differentiable. Losses can be used as metrics

## Loss functions - regression - MAE and MSE
>[!def] Mean Absolute Error

>[!def] Mean squared error

Look at the comparison of MAE and MSE on the lecture slides 

>[!def] Root Mean Square Error

## Outliers

- Analyse the data and decide well how and when to remove outliers
- Never leave this out 
- quality of data is very important

## Metrics - Regression
### $R^2$

- How does the model make predictions different from the ground truths
- its not r squared. its R two
- can be negative
- Have a look at the equation
- negative when you predict completely wrong.
- 1 is the highest score 
### Adjusted $R^2$

- r two should go down with the more features we introduce
- more values increase average of dataset
- can you tell you if the model is either very good or very bad. no inbetween
### Correlation Coefficient

- similar to R two but not the same thing
- measures linear relationship

### Summary

- 10m in human scale? -> bad
- 10m in space scale? -> very good
- some in llinear, some in square scale, some are more biased.
- R two non-bais in non -linear models

## Metrics - Classifications

### Accuracy

- If equal plus one, else zero. just count them bitch
- not for the regression because there are continuous values
- 0-1 or 0-100
- Top-K Accuracy
	- multiple classes at the same time. 
	- top 5 predictions, if any are correct, it is correct 
- Shortcoming
	- how many data points are correct but not what type of errors were made. false positives or false negatives
	- Have a look at false negatives and false positives
	- Imbalance
		- fraud detection model
		- 1 in a million
		- model trained on imbalanced data almost impossible to get it to 100% accuracy
		- to get 99%: f(x)=0
- Balanced accuracy
	- accuracy for each class separately, and then take that average.
	- (TP + TN) /2

### Confusion matrix

- not a metric per say. 
- looks at TP, TN, FP, FN
- most common in binary classification, there is an extension to multiclass with more confusion or something

### Precision and Recall

- Precision = false positives (minimise)
- Recall = false negatives (minimise)
- nice figure in slides

There is a table in the slides with all the binary classification metrics
- in binary, there are four types of mistakes T/F P/N
- Accuracy, precision or recall and four components of something else 

### F-Score

- $F_1$ Score
	- binary 
	- precision and recall
	- 0.5 for precision and 0.5 for recall
- $F_\beta$

## Probabilistic binary classifiers - threshold

- lots of yaps omg
- probability of one class being predicted
- default is 0.5. right in the middle of the binary classification

## Receiver-Operating Curve  (ROC)

- threshold, 0-1
- as we move the threshold, the recall and precision change
- student corrects prof. ha ha... dry af
- im tired.
- i want home
- Now hes yapping about area under the curve.
- AUROC HAAAAAaaaaa

# Main mention 

- use multiple metrics...

on page 36. its not `like linear regression`. `like logistic regression`

Look at slides. 37,38

-  motivate metrics

## Multi-label classification 

- 43
- optimal point moves to top right
- P-R curve
	- optimal point 1,1
	- AUPR

mimimimimimimim
- if you learn this you will pass the exam. yap yap yap
- looks at slides
- make notes
- be a good boi
- its late 

# Second half

## Overfitting

- To calculate the gap in data. overfitting measure$$L_{gap}=L_{val}-L_{train}$$
- as we decrease complexity the accuracy increases. 
- need to find a model in the "sweetspot" 
	- balance betweeen complexity and accuracy

## Underfitting

- Model too simple to catch features in the data
- high training loss

## Assesment 

- need to be able to predit

- okay, rob destracted me

## Components of Machine learning 

- representation
- evaluation
- optimization

## Model Complexity 

- need to take care to keep the model simplified enough to not overfit
- ![[Pasted image 20241126182338.png]]
- ![[Pasted image 20241126182409.png]]
- I had drugs haha
	- i think i missed quite a lot here

- #double_dipping is when you train and evaluate a model on the same dataset.
	- your model is not generalisable
## Typical Train/Val/Test Ratios

> #_SecondPass for assignment 2 we need to write about dataset splits. i generated some code from chatGPT to extract all that info out of the lecture slides. here is nice condensed summary on splitting data.
> [[Dataset Splitting]]

- easy way to solve only using the same train and test data from a dataset when split, trained, and evaluated. 
	- train the model in different variations of splitting the data to find iterations where the train and test set are the most independent

> #leakage is when testing data is not independent of  

Root mean squared

Search for most linearly independent data points in the dataset

I thought the rest of the lecture was really self explanatory from the slides. and im having a better time understanding and taking occational notes instead of detailed summaries of the content. im tired. 5 hours of sleep two days in a row and sport inbetewen kills me.

## Time dependency

- nesting cross validation 
- train chronologically more pieces of data on the model

# Questions
- criteria when choosing model
	- amount
	- kind of data
- you need keep in mind the metrics, validation methods, data, 
