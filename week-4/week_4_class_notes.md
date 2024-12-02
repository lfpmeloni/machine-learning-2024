# Week 4

## Day 16

### Model Voting

Model voting combines predictions from multiple models to enhance the accuracy and reliability of predictions. The fundamental idea is to integrate diverse model outputs to make a collective decision, often improving the performance over individual models.

Key Types of Voting:

- Hard Voting: Each model votes for a class label, and the majority class is selected.
- Soft Voting: Uses the predicted probabilities from models. The class with the highest average probability is chosen.

        from sklearn.ensemble import VotingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split

        # Load dataset
        X, y = load_iris(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Instantiate individual classifiers
        clf1 = LogisticRegression(random_state=42)
        clf2 = SVC(probability=True, random_state=42)
        clf3 = DecisionTreeClassifier(random_state=42)

        # Create a VotingClassifier with soft voting
        voting_clf = VotingClassifier(estimators=[
            ('lr', clf1), ('svc', clf2), ('dt', clf3)], voting='soft')
        voting_clf.fit(X_train, y_train)

        # Evaluate
        print(voting_clf.score(X_test, y_test))

### Random Forest

Random Forest is an ensemble method that constructs multiple decision trees during training and merges their outputs for prediction. It introduces randomness by:

- Using a random subset of features at each split.
- Bootstrapping the data (sampling with replacement).

Advantages:

- Reduces overfitting compared to single decision trees.
- Handles both classification and regression tasks well.
- Computes feature importance.

        from sklearn.ensemble import RandomForestClassifier

        # Initialize the classifier
        rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)

        # Train the model
        rf_clf.fit(X_train, y_train)

        # Predict and evaluate
        print(rf_clf.score(X_test, y_test))

        # Feature importance
        print(rf_clf.feature_importances_)

### Bagging

Bagging (Bootstrap Aggregating) improves model stability by training multiple base models on different bootstrap samples and averaging their predictions (for regression) or majority voting (for classification).

Characteristics:

- Reduces variance in predictions.
- Effective for high-variance models like decision trees.

        from sklearn.ensemble import BaggingClassifier
        from sklearn.tree import DecisionTreeClassifier

        # Bagging with decision trees
        bagging_clf = BaggingClassifier(
            base_estimator=DecisionTreeClassifier(), n_estimators=50, random_state=42)
        bagging_clf.fit(X_train, y_train)

        # Evaluation
        print(bagging_clf.score(X_test, y_test))

### Gradient Boosting

Gradient Boosting builds models sequentially, where each new model attempts to correct the errors of the previous ones. Unlike Random Forests, it focuses on optimizing a loss function rather than averaging model outputs.

Characteristics:

- High accuracy for both regression and classification tasks.
- Tends to be slower due to its sequential nature.

        from sklearn.ensemble import GradientBoostingClassifier

        # Gradient boosting
        gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
        gb_clf.fit(X_train, y_train)

        # Evaluation
        print(gb_clf.score(X_test, y_test))

### E-Scoring tag 15 Aufgabe
