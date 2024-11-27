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

SVC, SVR
