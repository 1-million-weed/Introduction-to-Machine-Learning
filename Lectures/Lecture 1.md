---
tags:
  - Marinus
  - Lectures
  - _FirstPass
  - Matthijs
---
# Course info
>[!note] Look at lecture slides for general info about the course. Very clear and structured

- Practicals are weekly starting week 2
- **Grade** is determined:
	- 50% homework x 3
		- first two are `theory`, last is fully `practical`
	- 50% exam
		- 40% multiple choice
		- 60% written

# Introduction
- Machine learning: algorithms learning from data
- types:
	- supervised
	- unsupervised
	- reinforced learning
- python libraries:
	- [Pandas](https://pandas.pydata.org/) - tabular data analysis tool.
	- [Pymarc](https://pypi.org/project/pymarc/) - for working with bibliographic data encoded in MARC21.
	- [Matplotlib](https://matplotlib.org/) - data visualization tools.
	- [BeautifulSoup](https://pypi.org/project/beautifulsoup4/) - for parsing HTML and XML documents.
	- [Requests](https://pypi.org/project/requests/) - for making HTTP requests (e.g., for web scraping, using APIs)
	- [Scikit-learn](https://scikit-learn.org/stable/) - machine learning tools for predictive data analysis.
	- [NumPy](https://numpy.org/) - numerical computing tools such as mathematical functions and random number generators.

# Lecture material
---
## What you need for AGI 

- [[Adaptivity & adaptability]]: learning from mistakes and improving
- [[Learning]]: acquiring new knowledge and understanding
- [[Generalization]]: Lessons learned should be general and widely applicable

## The qualification problem

It is really difficult to make a pre-condition list in real life.
Actions can have unintended consequences, enumerating these is also nearly impossible.
#_Err_Mismatch 
Work in specialised cases but not in anything where you didn't apply conditions

## Data driven Paradigms
There are many techniques combined in AI these days. Currently the dominant set in AI is ML.

## Learning paradigms

- ### Supervised learning 

	Supervised learning uses labelled data. The data is marked as e.g. correct or false, which the agent uses to learn.
	- It is limited by (labelled) data quantity
	- Producing labels is expensive

- ### Unsupervised learning

	Agent learns from data directly. There are no correct answers, no labels and thus no supervision.
	- It is not limited by labelling (unlike supervised learning), but is limited by how to capture additional data of interest.
	- The problem with this learning paradigm is algorithmic: it is hard to learn from unlabelled data and without reward systems
	- The data is broken down by the agent into different levels of features
		- e.g. a line, versus a simple shape, versus a complex shape, etc.

- ### Reinforced learning

	The agent interacts within an environment and uses a reward signal to learn.
	- The agent makes a policy to maximise the reward (from the reward signal)
	- This paradigm is commonly used for sequential decision making tasks

- ### Transfer learning

	Combines Reinforced and Unsupervised learning. The agent uses learned knowledge from previous tasks to improve at new tasks.
	- For example, in an image understanding problem: 
	  If one learns an image classifier for dogs/cats, the features that are learned in this case are also useful for classifying birds, as images naturally have features like edges, parts, etc.

- ### Combinations of learning paradigms

	These combine the aforementioned learning paradigms:
- **Self supervised**
- **Semi supervised**
- **Weak supervision**


### The cake / lecake
A representation of the relative values of learning paradigms {in eatable form, yummie}


## Basic ML concepts
- [[Model]]
- [[Parameters, weights]]
- [[Loss function]]
- [[Training data]]
- [[Classification]]
- [[Regression]]
- [[Multi task learning]]


# Challenges in Machine Learning

A canvas with this info can be found here: [[Exam Prep.canvas|Exam Prep]]
Or, here is a list of the challenges:
1. [[Quality of Data]]
2. [[Model Misspecification]]
3. [[Generalisation]]
4. [[Explainability]]
5. [[Bias, Safety, Trustworthiness]]


# Take Home Messages

- There are three paradigms in Machine Learning:
	- **Supervised,**
	- **Unsupervised,**
	- **and Reinforcement**.
- **[[Generalisation]]** is the biggest issue/core idea of Machine Learning.
- Two main tasks: 
	- **Classification**  
	- **Regression**
- Learning is done through an <mark style="background: #CACFD9A6;">optimization problem</mark>.
- **[[Normalisation]] and pre-processing** of features and labels is very important.
- **Machine Learning is not magic**; there are many challenges.


---
| Next: [[Lecture 2 Supervised Learning I - Linear Models]] |
