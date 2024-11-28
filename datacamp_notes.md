# DataCamp Courses

## Supervised Learning with scikit learn

## Content

1. k-Nearest Nighbors
2. Linear Regression
3. Logistic Regression
4. ROC Curve, Cross-Validation

## Understanding `scikit-learn`

        from sklearn.module import Model
        model = Model()
        model.fit(X, y)
        predictions = model.predict(X_new)
        print(predictions)

### k-Nearest Nighbors

Classifying labels of unseen data with k-Nearest Nighbors

Predict the label of a data point by k-Nearest Nighbors

        # Import KNeighborsClassifier
        from sklearn.neighbors import KNeighborsClassifier 

        y = churn_df["churn"].values
        X = churn_df[["account_length", "customer_service_calls"]].values

        # Create a KNN classifier with 6 neighbors
        knn = KNeighborsClassifier(n_neighbors=6)

        # Fit the classifier to the data
        knn.fit(X, y)

        # Predict the labels for the X_new
        y_pred = knn.predict(X_new)

        # Print the predictions
        print("Predictions: {}".format(y_pred)) 

#### Accuracy

Split data into Training set and test set -> Fit/train classifier on training set -> Calculate accuracy using test set: correct predictions / total observations

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21, stratify=y)
        knn = KNeighborsClassifier(n_neighbors=6)
        knn.fit(X_train, y_train)
        print(knn.score(X_test, y_test))

        # Model complexite and over/underfitting
        train_accuracies = {}
        test_accuracies = {}
        neighbors = np.arrange(1,26)
        for neighbor in neighbors:
                knn = KNeighborsClassifier(n_neighbors=neighbor)
                knn.fit(X_train, y_train)
                train_accuracies[neighbor] = knn.score(X_train, y_train)
                test_accuracies[neighbor] = knn.score(X_test, y_test)

        # Plot the results
        plt.figure(figsize=(8,6))
        plt.title("KNN: Varying Number of Neighbors")
        plt.plot(neighbors, train_accuracies.values(), label="Training Accuracy")
        plt.plot(Nighbors, test_accuracies.values(), label = "Testing Accuracy")
        plt.legend()
        plt.xlabel("Number of Neighbors")
        plt.ylabel("Accuracy")
        plt.show()

### Linear Regression

Supervised Learning with Scikit-Learn. Target value (label) typically have a continous value, such as a county's GDP, price of a house, etc.

    import pandas as pd
    diabetes_df = pd.read_csv("diabetes.csv")
    print(diabetes_df.head())

    X = diabetes_df.drop("glucose", axis=1).values
    y = diabetes_df["glucose"].values
    print(type(X), type(y))
    -> <class 'numpy.ndarray'> <class 'numpy.ndarray'>

    # Making predictions from a single feature
    X_bmi = X[:, 3]
    print(y.shape, X_bmi.shape)
    -> (752,) (752,)
    # Our feature must be at least 2 dimensional array to be accepted by scikit learn
    X_bmi = X_bmi.reshape(-1, 1)
    print(X_bmi.shape)
    -> (752, 1) # Correct shape for our model

    # Plotting glucose vs. body mass index
    import matplotlib.pyplot as plt
    plt.scatter(X_bmi, y)
    plt.ylabel("Blood Glucose (mg/dl)")
    plt.xlabel("Body Mass Index")
    plt.show()

    # Fitting a regression model (linear)
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression()
    reg.fit(X_bmi, y)
    prediction = reg.predict(X_bmi)
    plt.scatter(X_bmi, y)
    plt.plot(X_bmi, predictions)
    plt.ylabel("Blood Glucose (mg/dl)")
    plt.xlabel("Body Mass Index")
    plt.show()

Regression mechanics: fit the line y = ax + b minimizing the error using an error function, also known as loss or cost functions.

RSS: Residual sum of squares -> Ordinary Least Squares (OLS) to minimize RSS

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    reg_all = LinearRegression()
    reg_all.fit(X_train, y_train)
    y_rped=reg_all.predict(X_test)

The default metric for linear regression is R-squared quintifieng the ammount of variance in the target variable that is explained by the features. Range from [0,1]

    reg_all.score(X_test, y_test)

Mean of the residual sum of squares (MSE) Mean Squared Error.
RMSE = root(MSE) to keep the same unit as the target variable.

    from sklearn.metrics import mean_squared_error
    mean_squared_error(y_test, y_pred, squared=False)

Cross-validation: Model performance is dependent on the way we split up the data making it not representative  of the model's ability to generalize to unseen data. Cross-validation is a technique to combat this dependance to random split.
Split the data in X fold and keep one fold as test data and the rest as train. Iterate X times for each fold and store the metric for each, such as score.

    from sklearn.model_selection import cross_val_score, KFold
    kf = KFold(n_splits=6, shuffle=True, random_state=42)
    reg = LinearRegression()
    cv_results = cross_val_score(reg, X, y, cv=kf)
    print(cv_results)
    print(np.mean(cv_results), np.std(cv_results))
    print(np.quantile(cv_results, [0.025, 0.975])) # 95% confidence interval

#### Regularized regression (Regularzation for regression models)

Large coefficients can lead to overfitting so regularization penalizes large coefficients.

1. Ridge regression: Loss function = OLS loss function + alpha * each point squared. Alpha is a hyperparameter used to optimize model parameters and penalizes large positive or negative coefficients. alpha equal zero is performing OLS which can lead to overfitting and high alpha can lead to underfitting

    from sklearn.linear_model import Ridge
    scores = []
    for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]:
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train, y_train)
        y_pred = ridge.predict(X_test)
        scores.append(ridge.score(X_test, y_test))
    print(scores)

2. Lasso regression: OLS function plus the absolute value of each coefficient multiplied by some constant alpha. Can select important features of a dataset. Schrinks the coefficients of less important features to zero. Features that are not schrunk to zero are selected by lasso
    from sklearn.linear_model import Lasso
    scores = []
    for alpha in [0.1, 1.0, 10.0, 20.0, 50.0]:
        lasso = Lasso(alpha=alpha)
        lasso.fit(X_train, y_train)
        lasso_pred = lasso.predict(X_test)
        scores.append(lasso.score(X_test, y_test))
    print(scores)

### Lasso for feature selection in scikit-learn

    from sklearn.linear_model import Lasso
    X = diabetes_df.drop("glucose", axis=1).values
    y = diabetes_df["glucose"].values
    names = diabetes_df.drop("glucose", axis=1).columns
    lasso = Lasso(alpha=0.1)
    lasso_coef = lasso.fit(X, y).coef_ # Extract the coeficients from Lasso
    plt.bar(names, lasso_coef)
    plt.xticks(rotation=45)
    plt.show()

### Confusion matrix: How good is our model?

Accuraccy can not always be a good metric for our model. Class imbalance is when we have uneven frequency between classes (Ex: 99% non fraudulent transaction you can have 99% accuracy without getting one fraudulent transaction right). A confusion matrix can be used for assessing the classification performance. True positive, true negative, false negative and false positive.

Accuracy: Sum of true predictions devided by the total sum of matrix
Precision: True positives / (true positives + false positives) - positive predictive value
Recall: True positives / (true positives + false negatives) - sensitivity
F1 score: Harmonic mean of precision and recall: 2 *( precision* recall) / (precision + recall)

    from sklearn.metrics import classification_report, confusion_matrix
    knn = KNeighborsClassifier(n_neighbors=7)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=42)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

### Logistic Regression for binary classification

Used for classification problems. It outputs probabilities. If p > 0.5 the data is labeled as 1. It produces a linear decision boundary.

    from sklearn.linear_model import LogisticRegression
    logreg = LogisticRegression()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    # Predictiong probabilities
    y_pred_probs = logreg.predict_proba(X_test)[:, 1]
    print(y_pred_probs[0])
    -> [0.08961376]

By default the logistic regression threshold is 0.5. Not specific to logsitic regression, KNN also have thresholds.

### The ROC Curve

The receiver operating characteristics curve can show how different threascholds addect true positive and false positive rates.

    from sklearn.metrics import roc_curve
    # unpack results into false positive rate (fpr), true positive rate (tpr) and thresholds
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)
    plt.plot([0,1],[0,1],'k--')
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Logistic Regression ROC Curve')
    plt.show()

ROC AUC: by calculating the area under the ROC curve you obtain p which express the % of correctly predicting the outcome.

    from sklearn.metrics import roc_auc_score
    print(roc_auc_score(y_test, y_pred_probs))

### Choosing the correct hyperparameters

It is essential to use cross-validation to avoid overfitting to the test set.
Grid search cross validation: Can test different parameters while choosing also different metrics. It doesnt scale well since it iterates over hyperparameters, number of values and by the number of folds.

    from sklearn.model_selection import GridSearchCV
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    param_grid = {"alpha": np.arange(0.0001, 1, 10),"solver":["sag", "lsqr"]}
    ridge = Ridge()
    ridge_cv = GridSearchCV(ridge, param_grid, cv=kf)
    ridge_cv.fit(X_train, y_train)
    print(ridge_cv.best_params_, ridge_cv.best_score_)

#### RandomizedSearchCV

    from sklearn.model_selection import RandomizedSearchCV
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    param_grid = {'alpha': np.arange(0.0001, 1, 10), 'solver': ['sag', 'lsqr']}
    ridge = Ridge()
    ridge_cv = RandomizedSearchCV(ridge, param_grid, cv=kf, n_iter=2)
    ridge_cv.fit(X_train, y_train)
    print(ridge_cv.best_params_, ridge_cv.best_score_)
    test_score = ridge_cv.score(X_test, y_test)
    print(test_score)

## Preprocessing data

scikit-learn required numeric data and no missing values which are rare on real-world problems. We will often neet to preprocess our data first.

When dealing with categorical features, scikit-learn will not accept categorical features by default so we need to convert into numeric values which are called dummy variables. This can be achieved using scikit-learn OneHotEncoder() or pandas get_dummies().

    import pandas as pd
    music_df = pd.read_csv('music.csv')
    # As we only need 9 out of 10 possible features we can drop the first item
    music_dummies = pd.get_dummies(music_df["genre"], drop_first=True)
    print(music_dummies.head())
    # To bring these binary features back to original df use concat
    music_dummies = pd.concat([music_df, music_dummies], axis=1)
    # Now we can remove the original genre column using drop
    music_dummies = music_dummies.drop("genre", axis=1)
    # If the data frame only has one categorical feature we can jump to
    music_dummies = pd.get_dummies(music_df, drop_first=True)

### Linear regression with dummy variables

    from sklearn.model_selection import cross_cal_score, KFold
    from sklearn.linear_model import LinearRegression
    X = music_dummies.drop("popularity", axis = 1).values
    y = music_dummies["popularity"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    linreg = LinearRegression()
    # scoring using negative MSE because scikit understands that higher score is better so needs to couteract
    linreg_cv = cross_val_score(linreg, X_train, y_train, cv=kf, scoring = "neg_mean_squared_error")
    print(np.sqrt(-linreg_cv))

### Handling Missing Data

There might be no value for a feature in a particular row because there may have been no observation or the data might be corrupt

    # Show how many na vlaues on each column
    print(music_df.isna().sum().sort_values())

Possible approaches:

1. Remove missing observation accounting for less than 5% of all data

    music_df = music_df.dropna(subset=["genre", "popularity", "loudness", "liveness", "tempo"])
    print(music_df.isna().sum().sort_values())

2. Impute missing data using subject-matter expertise to replace with educated guesses. Can use the mean, median, or another value. For categorical values, we typically use the most frequent value - the mode. It is important to split our data first to avoid *data leakage* bringing test data set to our model. Imputers are also known as transformers

        from sklearn.impute import SimpleImputer
        # Since we are using different impute for cat an num sets, we split them
        X_cat = music_df["genre"].values.reshape(-1,1)
        X_num = music_df.drop(["genre", "popularity"], axis=1).values
        y = music_df["popularity"].values
        # Create categorical and numerical train and test sets (important to mantain the same random state)
        X_train_cat, X_test_cat, y_train, y_test = train_test_split(X_cat, y, test_size=0.2, random_state=12)
        X_train_num, X_test_num, y_train, y_test = train_test_split(X_num, y, test_size=0.2, random_state=12)
        imp_cat = SimpleImputer(strategy="most_frequent")
        X_train_cat = imp_cat.fit_transform(X_train_ cat)
        X_test_cat = imp_cat.transform(X_test_cat)
        # Instantiate another imputer for numerical values
        imp_num = SimpleImputer() # mean is dafault
        X_train_num = imp_num.fit_transform(X_test)
        X_test_num = imp_num.transform(X_test_num)
        # We then combine our training data using append()
        X_train = np.append(X_train_num, X_train_cat, axis=1)
        # Repeat for our test data
        X_test = np.append(X_test_num, X_test_cat, axis = 1)

#### Imputing within a pipeline

A pipeline is an object used to run a series of transformations and build a model in a single workflow.

    from sklearn.pipeline import Pipeline
    music_df = music_df.dropna(subset=["genre", "popularity", "loudness", "liveness", "tempo"])
    # We convert values in the genre column, ehich will be the target, to 1 if rock, else 0
    music_df["genre"] = np.where(music_df["genre"] == "Rock", 1, 0)
    X = music_df.drop("genre", axis=1).values
    y = music_df["genre"].values
    # define pipeline steps
    steps = [("imputation", SimpleImputer()), ("logistic_regression", LogisticRegression())]
    pipeline = Pipeline(steps)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)
    pipeline.fit(X_train, y_train)
    pipeline.score(X_test, y_test)
    # In a pipeline, each step but the last must be a transformer
    # If it was a knn method
    y_pred = pipeline.predict(X_test)

### Centering and scaling

Get a description of main parameters from our data, such as, count, mean, std, min, 25%, 50%, 75% and max.

    print(music_df[["duration_ms", "loudness", "Speechiness"]].describe())

Many models use some form of distance to inform and features on larger scales can disproportionately influence the model. KNN for example uses distance explicitly when making predictions. So we want features to be on a similar scale applying normalizing or standardizing (scalling and centering)

#### Standardization

Subtract the mean and divide by variance. All features are centered around zero and have variance of one.

    from sklearn.preprocessing import StandardScaler
    X = music_df.drop("genre", axis=1).values
    y = music_df["genre"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(np.mean(X), np.std(X))
    print(np.mean(X_train_scaled), np.std(X_train_scaled))

#### Normalization

1. Subtract the minimum and divide by the range so minimum zero and maximum one.
2. Normalize so the data ranges from -1 to +1
3. scikit-learn docs for other available types of scaling

#### Scaling in a pipeline

    steps = [('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors=6))]
    pipeline = Pipeline(steps)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)
    knn_scaled = pipeline.fit(X_train, y_train)
    y_pred = knn_scaled.predict(X_test)
    print(knn_scaled.score(X_test, y_test))
    
    #Comparing to a unscaled data
    knn_unscaled = KNeighborsClassifier(n_neighbors=6).fit(X_train, y_train)
    print(knn_unscaled.score(X_test, y_test))

#### Cross-validation and scaling in a pipeline

    from sklearn.model_selection import GridSearchCV
    steps = [('scaler', StandardScaler()), ('knn', KNeighborsClassifier())]
    pipeline = Pipeline(steps)
    # double underscore followed by the hyperparameter name
    parameters = {"knn__n_neighbors": np.arange(1, 50)}
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)
    cv = GridSearchCV(pipeline, param_grid=parameters)
    cv.fit(X_train, y_train)
    y_pred = cv.predict(X_test)
    print(cv.best_score_)

## OK, but how do we dicide which model to use in the first place?

Some guiding pinciples:

1. Size of the dataset: Fewer features = simpler model, faster training time. Some models require large amounts of data to perform well.
2. Interpretability: Some models are easier to explainm which can be important for stakeholders such as Linear Regression undertanding coefficients.
3. Flexibility: Generally, flexible models make fewer assumptions about the data which may improve accuracy. For example KNN is a more flexible model by not assuming any linear relationship.

It's all in the metrics. Regression model performance can be evaluated in terms of RMSE and R-aquared. Classification model performance can be evaluated in terms of Accuracy, Confusion matrix, Precision, recall, F1-score, ROC AUC and so on.

Therefore, one approach is to select several models and a metric then evaluate their performance without any form of hyperparameter tuning.

Note that there are models affected by scaling such as KNN, Linear Regression (plus Ridge, Lasso), Logistic Regression, Artificial Neural Network. Therefore, it is generally best to scale our data before evaluating models out of the box.

### Evaluating classification models

    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score, KFold, train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    X = music.drop("genre", axis=1).values
    y = music["genre"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # With our data scalled, then we describe a dictionary with our model names as strings for the keys and instantiate models as the dictionary's values
    models = {"Logistic Regression": LogisticRegression(), "KNN": KNeighborsClassifier(), "Decision Tree": DecisionTreeClassifier()}
    # Empty list to store the results
    results = []
    # Create a loop
    for model in models.values():
        kf = KFold(n_splits=6, random_state=42, shuffle=True)
        # By default the scoring metric here will be accuracy
        cv_results = cross_val_score(model, X_train_scaled, y_train, cv=kf)
        results.append(cv_results)
    # Visualize the range from the models results
    plt.boxplot(results, labels=models.keys())
    plt.show()

    # To evaluate on the test set we can loop through the names and values of the dictionary
    from name, model in models.items():
        # Fit the model, calculate accuracy and print it
        model.fit(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)
        print("{} Test Set Accuracy: {}". format(name, test_score))

## DataCamp Course 2 | Unsupervised Learning

Learning without labels or targets as to find patterns in data but without a specific prediction task in mind.

### Clustering

Iris dataset: Measurements of many iris plants
Three species: setosa, versicolor and virginica
Measurements: Petal length, petal width, sepal length, sepal width
Through this course datasets like this will be 2D NumPy array
4 dimensional dataset

#### k-means clustering

Finds clusters of samples, number of clusters must be specified and is implementend within the sklearn library.

    import pandas as pd

    # Load the dataset
    iris_ds = pd.read_csv("iris.csv")
    print(iris_ds.head())

    from sklearn.cluster import KMeans
    model = KMeans(n_clusters=3) # 3 types of iris
    model.fit(samples)

    labels = model.predict(samples)
    print(labels)

Cluster labels for new samples:

- New samples can be assigned to existing clusters.
- k-means remembers the mean of each cluster (the "centroids")
- New samples are assigned to the cluster whose centroid is closest

Supose we have have a new dataset of iris plants called new_samples, we could predict based on the already trained set.

    new_labels = model.predict(new_samples)
    print(new_labels)

It is quite usefull to use Scatter plots to view the real and predicted data sets.

    import matplotlib.pyplot as plt
    xs=samples[:,0]
    ys=samples[:,2]
    plt.scatter(xs,ys, c=labels)

Evaluating a clustering: Check for correspondences, measure the quality of a clustering, inform choice of how many clusters to look for. It is usefull to work with *cross-tabulation* within labels found by algorithm and pre-known target.

    import pandas as pd
    df = pd.DataFrame({'lables': labels 'species': species})
    print(df)
    ct = pd.crosstab(df['labels'], df['species'])
    print(ct)

    # On my code
    df = pd.DataFrame({'labels':labels, 'species':iris_ds["class"].values})
    print(df)

Cross tabulation like these provide great insight into which sort of samples are in which cluster.

How can the quality of the clustering be evaluated when you dont have the label of the data such as the flower species? How spread out the samples within each cluster can be measured by the "*inertia*" where lower is better. *Distance* from each sample to centroid of its cluster. The inertia is calculated every fit and is avalilable as an attribute from the model. KMeans aims to place the clusters in a way that minimizes the inertia.

    from sklearn.cluster import KMeans
    model=KMeans(n_clusters=3)
    model.fit(sample)
    print(model.inertia_)

A good clustering has tight clusters (so low inertia) but not too many clusters. Choosing should be made at an "elbow" in the inertia plot where inertia begins to decrease more slowly.

    ks = range(1,11)
    inertias = []

    for k in ks:
        model = KMeans(n_clusters=k)
        model.fit(samples)
        inertias.append(model.inertia_)

    # Plot ks vs inertias
    plt.plot(ks, inertias, '-o')
    plt.xlabel('number of clusters, k')
    plt.ylabel('inertia')
    plt.xticks(ks)
    plt.show()

##### Transforming features for better clusterings

    from sklearn.cluster import KMeans
    model = KMeans(n_clusters=3)
    labels = model.fit_predict(samples)

    df = pd.DataFrame({'labels': labels, 'varieties': varieties})
    ct = pd.crosstab(df['labels'], df['varieties'])
    print(ct)

We can note that this time the KMenas with 3 clusters as our data set did not correspond well with the wine varieties. Thats because the features of the wine dataset have very different variances.

In KMeans features variance = feature influence. Therefore it might be usefull to have a mean 0 and variance 1 using StandardScaler using standardized features to train the model.

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(samples)
    StandardScaler(copy=True, with_mean=True, with_std=True)
    samples_scaled = scaler.transform(samples)

    from sklearn.pipeline import make_pipeline
    pipeline = make_pipeline(scaler,model)
    pipeline.fit(samples_scaled)
    labels = pipeline.predict(samples_scaled)

Besides 'StandardScaler', there are other preprocessing steps such as 'MaxAbsScaler', 'Normalizer'.

### Visualizing hierarchies

Unsupervised learning techniques for visualization and communicating insight: t-SNE and hierarchical clustering.

#### Hierarchical clustering

Example using the Eurovision 2016 countries scores for songs. Generates a 2D array of scores where rows are countries and columns are songs.

The Hierarchical Clustering of voting countries in a tree-like diagram called a "dendrogram". Relations can be found from closer to each other, geopolitical ties or same language.

Agglomerative clustering: Every country begins in a separate cluster, at each step, the two closest clusters are merged. It is repeated until only one cluster is left containing all the countries.

Devisive clustering works the other way around.

    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import linkage, dendrogram
    mergings = linkage(samples, method='complete')
    dendrogram(mergings, labels=country_names, leaf_rotation=90, leaf_font_size=6)
    plt.show()

An intermediate stage in the hierarchical clustering is specified by choosing a height on the dendrogram. Distance in the merging clusters is represented by the hight.

Linkage method, for example, complete measures distance between clusters is max distance between their samples.

Extracting cluster labels: We can use the fcluster() function which returns a NumPy array of cluster labels.

    from scipy.cluster.hierarchy import linkage, dendrogram
    mergings = linkage(samples, method='complete')
    from scipy.cluster.hierearchy import fcluster
    labels = fcluster(mergings, 15, criterion='distance')
    print(labels)

To inspect the cluster labels, lets use a DataFrame to align the labels with the country names.

    import pandas as pd
    pairs = pd.DataFrame({'labels': labels, 'countries': country_names})
    print(pairs.sort_values('labels'))

#### t-SNE (i-distributed stochastic neighbor embedding)

Creates a 2D map of a dataset conveys useful information about the proximity of the samples to one another. It maps samples from high-dimensional space into a 2 or 3 dimensional space so they can be visualized. Some distortion is inevitable but this methos is able to bring a visual aid for understanding a dataset.

    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    model = TSNE(learning_rate=100)
    transformed = model.fit_transform(samples)
    xs = transformed[:,0]
    ys = transformed[:,1]
    plt.scatter(xs, ys, c=species)

t-SNE only has a `fit_transfrom()` method and can't be extended to include new data samples. Usually when you choose the wrong learning rate the point will be bunched together. Normal values are between 50 and 200. Also the axes of the t-SNE plot do not have any interpretable meaning and will be different each time tSNE is applied even on the same data.

#### The PCA transformation

Techniques for dimension reduction. It finds patterns in data and uses these patterns to re-express it in a compressed form. Subsequent computation can be much more efficient and is a big deal in world of big datasets. Remove less-informative 'noise' features whicg cause problem for prediction tasks.

PCA - Principal Component Analysis

Fundamental dimension reduction technique. First step is decorrelation and the second is reducing dimensions. PCA aligns data with axes by rotating them and shifts them to have mean 0.

    from sklearn.decomposition import PCA
    model = PCA()
    model.fit(samples)
    transformed = model.transformed(samples)
    print(transformed)

Rows of transformed correspond to samples. Columns of transformed are the "PCA features". Row gives PCA feature values of corresponding sample. Features of the dataset are often correlated but by performing the rotation it can decorrelate it making the columns of the transformed array not linearly correlated. The principal components are the directions of greater variance and are to those vectors that the PCA alings to the coordinate axis.

    print(model.components_) # Attribute generated after the fit with one row for each principal component.

In multidimensional feature space, the PCA features are ordered by variance descending. Using the PCA on Iris dataset, for example, shows that only 2 PCA features have bigger variance. *Intrinsic dimension is the number of PCA features that have significant variance.*

    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    pca = PCA()
    pca.fit(samples)
    features = range(pca.n_components_)

    # Make a barplot of the variances
    plt.bar(features, pca.explained_variance_)
    plt.xticks(features)
    plt.ylabel('variance')
    plt.xlabel('PCA feature')
    plt.show()

Intrinsic dimension can be ambiguous as it is an idealization. Then we can use dimension reduction.

### Dimension reduction with PCA

Represent the same data using less features. Low variance features after appling PCA can be assumed as "noise" retaining the high variance features which are informative.

The number of components to keep on the PCA must be informed.

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca.fit(samples)
    transformed = pca.transform(samples)
    print(transformed.shape)

    import matplotlib.pyplot as plt
    xs = transformed[:,0]
    ys = transformed[:,1]
    plt.scatter(xs, ys, c = species)
    plt.show()

Note that it can not always hold informative information of real world problem.

Word frequency arrays dataset: each row represents documents, columns represent words. Entries measure word-frequency - how often each word appears in each document. Matrices like these are sparce as most entries are zero and often represented using a special type of array called "csr_matrix" `scipy.sparse.csr_matrix` instead of NumPy array. This saves space storing only the non zeroes entries.

scikit-learn `PCA` doesn't support `csr_matrix`. Instead we can use `TruncatedSVD` instead which performs the same transformation.

    from sklearn.decomposition import TruncatedSVD
    model = TruncatedSVD(n_components=3)
    model.fit(documents) # Where documents is a csr_matrix
    trnasformed = model.transform(documents)

### Non-negative matrix factorization (NMF)

Its also a dimension reduction technique but unlike PCA, they are interpretable. Easier to interpret and to explain. They require all samples to be non-negative (>=0). It achieves its interpretability by decomposing samples as sums of their parts. For example combination of common themes in text or combination of common patterns on images.

It follow the same `fit()` `method()` pattern. The desired number of components `n_components` must always be specified. Works both with NumPy arrays and sparse arrays in the csr_matrix format. "tf" is the frequency of the word in document.

    from sklearn.decomposition import NMF
    model = NMF(n_components=2)
    model.fit(samples)
    nmf_features = model.transform(samples)
    print(model.components_)
    print(nmf_features) # will always be non negative

As with the PCA, NMF has components and just like PCA it has principal components that can be accessed from the model. The entries on a NMF component are always non negative.

Reconstruct of a samples: Features and components of an NMF model can be combined to aproximately reconstruct the original data samples. Product of matrices which corresponds to the Matrix Factorization in NMF.

    # Using NMF to scientific articles dataset
    print(articles.shape) -> (20000, 800)
    from sklearn.decomposition import NMF
    nmf = NMF(n_components=10)
    nmf.fit(articles)
    print(nmf.components_.shape) -> (10,800)

    df = pd.DataFrame(nmf_features, index=titles)
    print(df.loc['Anne Hathaway'])

NMF components represent topics and NMF features combine topics into documents but we can also use it on images as follows:

    print(sample) -> array of numbers
    bitmap = sample.reshape((2,3))
    print(bitmap)
    from matplotlib import pyplot as plt
    plt.imshow(bitmap, cmap='gray', interpolation='nearest')
    plt.show()

The course example for image is a digital digits number image and when we apply the NMC we can notice that each of the components are a segment of the digit and the features are which segments are part of each digit/number.

    def show_as_image(sample):
    bitmap = sample.reshape((13, 8))
    plt.figure()
    plt.imshow(bitmap, cmap='gray', interpolation='nearest')
    plt.colorbar()
    plt.show()

    # Import NMF
    from sklearn.decomposition import NMF
    model = NMF(n_components=7)
    features = model.fit_transform(samples)
    for component in model.components_:
        show_as_image(component)
    digit_features = features[0,:]
    print(digit_features)

NMF can be used to build a recommender system for say a similar article to read.

    from sklearn.decomposition import NMF
    nmf = NMF(n_components=6)
    nmf_features = nmf.fit_transform(articles)

#### Cosine similarity

Similar articles from same news writen differently can have diferent size in the same component. We can compare the angle between two lines to reach how the deviate one from another. Higher values indicate greater similarity.

    from sklearn.preprocessing import normalize
    norm_features = normalize(nmf_features)
    current_article = norm_features[23,:]
    similarities = norm_features.dot(current_article)
    print(similarities)

    import pandas as pd
    norm_features = normalize(nmf_features)
    df = pd.DataFrame(norm_features, index=titles)
    current_article = df.loc['Dog bites man']
    similarities = df.dot(current_article)
    print(similarities.nlargest())

## DataCamp Course 3 | Introduction to Deep Learning with PyTorch

What is deep learning? Traditional machine learning relies on hand-crafted feature engineering. Deep learning on the other hand, is able to discover features from raw data, giving them that edge over traditional machine learning.

Deep learning is beeign used for example in:

- Language translation
- Self-driving cars
- Medical diagnostics
- Chatbots

Deep Learning is contained within what is known as Machine Learning and basically contains an Input, Hidden Layers and Outputs having one or many hidden layers. It was inspired on neurons creating neural networks in the human brain. These models usually require a large amount of data.

In this course we are using PyTorch which is one of the most popular deep learning frameworks. It is used in industry as well among researchers. It has some common ground with the NumPy library.

### Tensor

Is similar to array which supports many mathematical operations and will form a cuilding block for our neural networks. Tensors can be created from Python lists by using the torch.tensor() class. They are similar to array and are the building blocks of neural networks.

    import torch
    lst = [[1, 2, 3], [4, 5, 6]]
    tensor = torch.tensor(lst)

PyTorch also supports tensor creation directly from NumPy arrays, using:

    np_array = np.array(array)
    np_tensor = torch.from_numpy(np_array)

Tensors are multidimensional representing a collection of elements arranged in a grid with multiple dimensions. Usefull functions:

    tensor.shape
    tensor.dtype
    device(type='cpu') # Deep learning often requires a GPU that offers parallel computing with faster training time and better performance

By running addition and subtraction is important to make sure that the shapes are compatible otherwise we get an error.

    a = torch.tensor([[1,1],[2,2]])
    b = torch.tensor([[2,2],[3,3]])
    a + b -> tensor([[3,3],[5,5]])
    a * b -> tensor([[2,2],[6,6]])

There are other common operations such as: `transposition`, `matrix multiplication` and `concatenation`. Most operations available on NumPy array operations can also be performed on PyTorch tensors.
