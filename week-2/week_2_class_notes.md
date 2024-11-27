# Week 2

## Day 6

Content of the week

- Bayes
- Linear Regression
- K Next Neighbors
- Decision Tree

### Bayes Classificator

Grundbegriffe der Statistik

Understand probability with and or groups

## Day 7

Studing Kapitel 4 from Book

Chapter 4 of the book Maschinelles Lernen – Grundlagen und Algorithmen in Python by Jörg Frochte introduces several foundational statistical concepts essential for machine learning, especially in relation to the Bayes classifier. Here’s a summarized version of the key points from the chapter and an explanation of the "fit" functionality, often seen in machine learning algorithms like those in scikit-learn.

### Statistical Basic Concepts

Chapter 4 lays out some key statistical concepts that form the foundation of machine learning algorithms:

Probability Theory: Used to quantify uncertainty in predictions. It provides the tools to model situations where outcomes are not deterministic.
Random Variables and Distributions: Describes the uncertainty in the system. For instance, a random variable represents possible outcomes of an experiment.
Expectation and Variance: Expectation is the average value a random variable takes, while variance measures how spread out the values are.
Bayes' Theorem: A core principle in machine learning. It helps update the probability estimate for a hypothesis as more evidence or data becomes available.
These concepts are crucial for understanding models like the Bayes classifier, which relies on probability distributions to make predictions.

### The "Fit" Functionality

In machine learning, the fit function is used to train a model on a dataset. Here's a general breakdown of its role:

Training the Model: When we call fit on a model (such as in scikit-learn), we provide the algorithm with training data (inputs and target values). The model "learns" from this data, adjusting its internal parameters to best capture the relationship between the inputs (features) and outputs (targets).

Example:

    from sklearn.naive_bayes import GaussianNB
    model = GaussianNB()
    model.fit(X_train, y_train)

In this example, X_train is the feature matrix, and y_train is the target vector. The model will adjust its parameters to maximize the likelihood of observing y_train given X_train.

The Learning Process: During the fit process, the model might use different methods depending on the algorithm. For instance, linear models use optimization techniques to minimize the error (loss function), while tree-based models recursively split data into subsets based on feature values.

### Learning Process - Overview

The general machine learning process, broken down for Chapter 4:

Data Collection: Gather a dataset containing features and target variables.
Data Preprocessing: Clean the data (handle missing values, outliers, etc.).
Model Selection: Choose the appropriate model (e.g., Naive Bayes, Decision Tree, SVM).
Model Training: Use the fit function to train the model on the training data.
Evaluation: Evaluate the model’s performance using test data, applying metrics like accuracy, precision, recall, etc.
Prediction: After training, use the model to make predictions on unseen data using the predict function.

### Bayesian Inference in Machine Learning

The Bayes classifier, a key model discussed in the chapter, applies Bayes’ Theorem to classify data. The algorithm calculates the posterior probability of different classes given the features of the data.

Key Steps in Naive Bayes Classifier:

Likelihood: Estimate the likelihood of the features given each class.
Prior: Assume a prior probability distribution of the classes before seeing the data.
Posterior: Update the prior with the likelihood to obtain the posterior.
The Naive Bayes classifier makes the naive assumption that features are conditionally independent given the class. This simplification allows for efficient calculation.

### Practical Example (Gaussian Naive Bayes)

from sklearn.naive_bayes import GaussianNB
import numpy as np

    # Example data
    X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])  # Features
    y_train = np.array([0, 1, 0, 1])  # Target labels

    # Model initialization and fitting
    model = GaussianNB()
    model.fit(X_train, y_train)

    # Making predictions
    y_pred = model.predict([[2, 3], [3, 4]])
    print("Predictions:", y_pred)

In this example, the fit method allows the Gaussian Naive Bayes model to learn the distribution of the features and labels. Afterward, predictions can be made on new data.

### Conclusion

Chapter 4 of Frochte's book provides a clear foundation for understanding the statistical concepts behind machine learning algorithms. The fit function is essential in the learning process, allowing models to adjust their parameters based on the training data. In particular, models like Naive Bayes, which are grounded in Bayes' Theorem, use the fit function to estimate the necessary parameters for making predictions.

## Day 8

Feiertag

## Day 9

### Linear regression

Linear regression models the relationship between one independent variable 𝑥 and a dependent variable 𝑦 using a straight line:

y=ax+b

Where:
a: Slope of the line (determines the steepness)
b: Intercept (where the line crosses the y-axis)

### Mehrdimensionale Regression - Multivariate Regression

Extends linear regression to multiple independent variables 𝑥1,𝑥2,...,𝑥𝑛

𝑦=𝑎1𝑥1+𝑎2𝑥2+⋯+𝑎𝑛𝑥𝑛+𝑏

Where:
𝑎1,𝑎2,...,𝑎𝑛a: Coefficients for each variable
𝑏: Intercept

### Polynomial Regression

Polynomial regression models relationships that are not linear by adding powers of 𝑥:

𝑦=𝑎0𝑥^0+𝑎1𝑥^1+𝑎2𝑥^2+⋯+𝑎𝑛𝑥^𝑛

Where:
𝑎0,𝑎1,...,𝑎𝑛: Coefficients for each polynomial term
𝑛: Degree of the polynomial

## Day 10

### Decision Trees

Decision trees split data recursively into branches based on features to reach predictions:

1. Root Node: The starting point of the tree.
2. Internal Nodes: Points where data is split based on a feature and condition.
3. Leaf Nodes: Endpoints that contain the prediction or output.

Example formula for splitting:

Gini Impurity=1−∑𝑖=1𝐶𝑝𝑖2

Where:
𝐶: Number of classes
𝑝𝑖: Proportion of samples in class 𝑖
