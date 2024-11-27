# Week 3

## Day 11

### Last weeks models review

- Bayes: Klassification: Arbeitet mit bedingten Wahrscheinlichkeiten.
- Lineare Regression: Regression oder Klassification mit LogisticRegression: Lineare Funktion. Training alle punkte sollen minimales abstand zur geraden haben. Voraussage mit Hilfe einer Geraden/Ebene/Hyperbolen
- DecisionTree: Klassification und Regression: Konstruiert Baum: Daten werden in Untergruppen geteilt, die immer "einheitlicher"

Ausreisser - outliers
Rauschen - noise

### K Neighbors (datacamp class)

Note: There is an imputer specifically for KNN - KNNImputer

### Regression Model Assumptions

When teaching regression models, it's common to mention the various assumptions underpinning linear regression. For completion, we'll list some of those assumptions here. However, in the context of machine learning we care most about if the predictions made from our model generalize well to unseen data. We'll use our model if it generalizes well even if it violates statistical assumptions. Still, no treatment of regression is complete without mentioning the assumptions.

- Validity: Does the data we're modeling matches to the problem we're actually trying to solve?
- Representativeness: Is the sample data used to train the regression model representative of the population to which it will be applied?
- Additivity and Linearity: The deterministic component of a regression model is a linear function of the separate predictors
- Independence of Errors: The errors from our model are independent.
- Homoscedasticity: The errors from our model have equal variance.
- Normality of Errors: The errors from our model are normally distributed.

When Assumptions Fail?
What should we do if the assumptions for our regression model aren't met? Don't fret, it's not the end of the world! First, double-check that the assumptions even matter in the first place: if the predictions made from our model generalize well to unseen data, and our task is to create a model that generalizes well, then we're probably fine. If not, figure out which assumption is being violated, and how to address it! This will change depending on the assumption being violated, but in general, one can attempt to extend the model, accompany new data, transform the existing data, or some combination thereof. If a model transformation is unfit, perhaps the application (or research question) can be changed or restricted to better align with the data. In practice, some combination of the above will usually suffice.

## Day 12

### Support Vector Machine (SVM)

#### Introduction to SVMs

Concept: SVM aims to find the optimal hyperplane that maximizes the margin between two classes in a dataset.

The hyperplane equation: 𝑔(𝑥)=𝑤0+𝑤1𝑥1+𝑤2𝑥2+…+𝑤𝑛𝑥𝑛

The margin is the distance between the closest data points (support vectors) and the hyperplane​.

Soft Margin for Non-Linearly Separable Classes:

- Introduces slack variables 𝜉𝑖 to allow misclassifications for better generalization​.
- Controlled by the 𝐶 parameter: Higher values prioritize correct classification, while lower values focus on maximizing the margin​​.

Kernel Tricks

To handle non-linear data, SVM uses kernel functions to transform data into higher-dimensional spaces​.

Common kernels:

- Linear: Used when data is linearly separable.
- Polynomial: Can capture interactions of features up to a certain degree​.
- Radial Basis Function (RBF): Adds flexibility to separate data that isn’t linearly separable in lower dimensions​.

Using scikit-learn
SVMs are implemented in scikit-learn as SVC (Support Vector Classifier) and LinearSVC for linear kernels​​.

    from sklearn import svm
    model = svm.SVC(kernel='rbf', C=1, gamma='auto')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

Adjust parameters such as C, gamma (for RBF kernel), and degree (for polynomial kernel) to optimize performance.

## Day 13

### Neural Networks

#### Introduction to Neural Networks

Neural networks are inspired by biological neurons, designed to recognize patterns through layers of interconnected nodes (neurons).

Perceptron: Simplest type of neural network, representing a single layer with a step activation function.
Limitations: Cannot solve non-linear problems like XOR without transformation.

Multi-Layer Perceptron (MLP): Overcomes limitations using multiple layers and non-linear activation functions.

#### Key Concepts

Activation Functions: Transform the input signals to introduce non-linearity.
Examples: ReLU, sigmoid, tanh.

Loss Functions: Measure the error between predictions and actual outcomes.
Examples: Mean Squared Error (MSE), Cross-Entropy Loss.

Optimization: Uses algorithms like Stochastic Gradient Descent (SGD) to minimize the loss.

#### Training Neural Networks

Steps:

1. Initialize weights randomly.
2. Forward propagation to compute outputs.
3. Compute loss.
4. Backpropagation to update weights using gradient descent.
5. Iterate until convergence.

Overfitting Solutions:

- Regularization (L1/L2 norms).
- Dropout (randomly deactivating neurons during training).
- Data augmentation for larger training sets.

#### Implementation in scikit-learn

    from sklearn.neural_network import MLPClassifier
    model = MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=500)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
