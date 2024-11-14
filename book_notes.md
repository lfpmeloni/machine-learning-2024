# "Maschinelles Lernen - Grundlagen und Algorithmen in Python" by Jörg Frochte (2021)

## Chapter 1: Introduction

Overview: Introduction to machine learning, its importance, and common use cases.
Code Example: Basic setup and installation of required libraries.

## Chapter 2: Machine Learning – Overview and Distinctions

2.1 What Does Learning Mean?: Definition and types of learning.
2.2 AI, Data Mining, and Knowledge Discovery: Differentiates AI from machine learning and data mining.
2.4 Supervised, Unsupervised, and Reinforcement Learning: Overview of learning paradigms with real-world examples.
Code Example: Demonstrates setting up a simple supervised learning pipeline​(Jörg Frochte - Maschine…).

## Chapter 3: Python, NumPy, SciPy, and Matplotlib – In a Nutshell

3.1 Installation with Anaconda and Spyder: Basic environment setup.
3.3 Matrices and Arrays in NumPy: Array manipulation and matrix operations.
3.6 Visualization with Matplotlib: Creating plots and visualizations.
Code Example:

        import numpy as np
        import matplotlib.pyplot as plt

        # Sample data and plotting
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        plt.plot(x, y)
        plt.show()

## Chapter 4: Statistical Foundations and Bayes Classifier

4.1 Statistical Basics: Covers key statistical concepts used in ML.
4.2 Bayes’ Theorem and Scale Levels: Explanation of Bayes’ theorem.
4.3 Bayes Classifier: Introduction and implementation of the Bayes classifier.
Code Example: Implementation of Bayes theorem for simple classification tasks​(Jörg Frochte - Maschine…).

## Chapter 5: Linear Models and Lazy Learning

5.1 Vector Spaces, Metrics, and Norms: Fundamental vector and norm concepts.
5.2 Least Squares Method for Regression: Explanation of regression analysis.
5.4 k-Nearest-Neighbor Algorithm (k-NN): Lazy learning techniques.
Code Example:

        from sklearn.neighbors import KNeighborsClassifier

        # Simple k-NN example
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_train, y_train)
        predictions = knn.predict(X_test)

## Chapter 6: Decision Trees

6.1 Trees as Data Structure: Introduction to tree structures.
6.2 Classification Trees with ID3 Algorithm: Basics of ID3 algorithm.
6.4 Overfitting and Pruning: Techniques for controlling model complexity.
Code Example: ID3 algorithm-based decision tree creation​(Jörg Frochte - Maschine…).

## Chapter 7: Single- and Multi-layered Feedforward Networks

7.1 Single-layer Perceptron and Hebbian Learning Rule: Basics of neural networks.
7.2 Multilayer Perceptron and Gradient Descent: Deep learning foundations.
Code Example:

        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense

        # Basic neural network with Keras
        model = Sequential([
            Dense(64, activation='relu', input_shape=(input_dim,)),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy')

## Chapter 8: Deep Neural Networks with Keras

8.1 Sequential Model of Keras: Introduction to Keras models.
8.5 Overfitting and Regularization: Techniques to improve model generalization.
Code Example: Implementing batch normalization and dropout to prevent overfitting.

## Chapter 9: Feature Engineering and Data Analysis

9.1 Pandas in a Nutshell: Data manipulation with pandas.
9.3 Feature Selection and Principal Component Analysis (PCA): Dimensionality reduction.
Code Example: PCA for feature reduction.

        from sklearn.decomposition import PCA

        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(X)

## Chapter 10: Ensemble Learning with Bagging and Boosting

10.1 Bagging and Random Forest: Introduction to ensemble methods.
10.3 Gradient Boosting: Advanced boosting techniques.
Code Example: Random forest classifier.

        from sklearn.ensemble import RandomForestClassifier

        rf = RandomForestClassifier(n_estimators=100)
        rf.fit(X_train, y_train)
        predictions = rf.predict(X_test)

## Chapter 11: Convolutional Neural Networks with Keras

11.1 CNN Basics: Understanding convolutions for images.
11.5 Transfer Learning: Applying pre-trained models.
Code Example: Simple CNN with Keras​(Jörg Frochte - Maschine…).

## Chapter 12: Support Vector Machines

12.1 Optimal Separation: Theory behind SVMs.
12.4 SVM in Scikit-Learn: Practical SVM with scikit-learn.
Code Example:

        from sklearn.svm import SVC

        svc = SVC(kernel='linear')
        svc.fit(X_train, y_train)
        predictions = svc.predict(X_test)

## Chapter 13: Clustering Methods

13.1 k-Means and k-Means++: Clustering basics and improvements.
13.3 DBSCAN for Density-Based Clustering: Advanced clustering.
Code Example: Implementing k-Means in Python.

        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=3)
        kmeans.fit(X)
        labels = kmeans.predict(X)

## Chapter 14: Fundamentals of Reinforcement Learning

14.1 Software Agents and Environment: Introduction to RL.
14.3 Q-Learning: Basic Q-learning algorithm.
Code Example: Q-learning for simple RL tasks.

## Chapter 15: Advanced Reinforcement Learning Topics

15.1 Experience Replay and Batch RL: Enhancements in RL.
15.9 Multi-Agent Scenarios: Advanced multi-agent interactions.
Code Example: Double Q-Learning implementation​(Jörg Frochte - Maschine…).
