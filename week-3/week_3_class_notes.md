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

## Day 14

### Sample code from the teacher

    Normale Daten
    nn = MLPRegressor(hidden_layer_sizes = (3,4),
    activation = "relu",
    solver = "adam",
    max_iter=2_000,
    random_state = 42)
    Große Daten
    nn = MLPRegressor(hidden_layer_sizes = (300,300,200,2),
    activation = "tanh",
    solver = "sgd",
    learning_rate= "invscaling",
    batch_size =1000,
    max_iter=2_0000,
    random_state = 42,
    warm_start = True,
    early_stopping = True)

### PCA - Principal Component Analysis

#### Introduction to PCA

PCA, or Principal Component Analysis, is a method for dimensionality reduction in data preprocessing.
Goal: Transform a high-dimensional feature space into a smaller one while retaining as much variance (information) as possible​​.

#### Key Concepts for PCA

(1) Projection to a Lower Dimensional Space:

- PCA reduces dimensions by projecting data points onto a new set of orthogonal axes (principal components).
- The first principal component captures the highest variance, followed by the second, and so on​​.

(2) Mathematical Foundation:

- PCA finds the eigenvalues and eigenvectors of the covariance matrix of the dataset.
- The eigenvectors define the direction of the new feature axes, and eigenvalues measure the variance along these axes​.

(3) Linear Transformation:

- PCA constructs a projection matrix 𝑊 from the top 𝑘 eigenvectors. The data is transformed as 𝑍 = 𝑊^𝑇𝑋, where 𝑋 is the original dataset​.

(4) Standardization:

- Standardizing the dataset (mean=0, variance=1) is essential to ensure fair treatment of all features​.

#### When to Use PCA

- Ideal for datasets with linearly correlated features.
- Useful in preprocessing to reduce overfitting in machine learning models​.
- Ineffective if features are non-linearly dependent—consider autoencoders or t-SNE for such cases​.

#### Steps in PCA

(1) Standardize the Data:

- Center the data around zero mean and unit variance.

    from sklearn.preprocessing import StandardScaler
    X_scaled = StandardScaler().fit_transform(X)

(2) Compute the Covariance Matrix:

- Reflects the relationships between different dimensions.

(3) Calculate Eigenvalues and Eigenvectors:

- Eigenvalues define the amount of variance captured by each principal component.
- Eigenvectors determine the direction of the new axes.

(4) Select Principal Components:

- Choose the top 𝑘 components based on eigenvalues.

(5) Project the Data:

- Transform the data onto the selected principal components.

#### Using PCA with scikit-learn

    from sklearn.decomposition import PCA

    # Fit PCA model with 2 components
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Explained variance ratio
    print(pca.explained_variance_ratio_)

Key Parameters:

- n_components: Number of dimensions to retain
- explained_variance_ratio_: Shows how much variance each principal component captures.
