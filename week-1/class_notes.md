# Week 1

## Day 1

### Understanding Machine Intelligence

#### What Machines Can Do

- Data Processing: Handle large datasets quickly (e.g., data analysis, pattern recognition).
- Automation: Perform repetitive tasks efficiently (e.g., manufacturing robots, automated customer service).
- Prediction: Forecast outcomes based on historical data (e.g., weather forecasting, stock market trends).
- Image and Speech Recognition: Identify objects, faces, and transcribe spoken language (e.g., facial unlock on smartphones, virtual assistants).

#### What Machines Cannot Do

- Creativity: Generate truly original ideas or art without human input.
- Emotional Understanding: Experience or comprehend emotions authentically.
- Common Sense Reasoning: Make judgments based on everyday knowledge and experiences.
- Contextual Understanding: Grasp nuanced meanings in complex situations.

#### Are Machines Really Intelligent?

- Narrow Intelligence: Machines excel in specific tasks but lack general intelligence.
- Lack of Consciousness: No self-awareness or subjective experiences.
- Dependence on Data: Intelligence is limited to the quality and scope of the data they are trained on.

#### What Humans Do Better

- Emotional Intelligence: Understand and manage emotions in themselves and others.
- Creative Problem-Solving: Innovate and think outside the box.
- Ethical Decision-Making: Make moral judgments considering complex societal values.
- Adaptability: Quickly adjust to new and unforeseen circumstances.

#### When Machines Deceived Humanity

- Deepfakes: AI-generated fake videos that can mislead people (e.g., Deepfake Example).
- AI Bias: Algorithms that unintentionally perpetuate biases present in training data (e.g., biased hiring tools).
- Autonomous Errors: Self-driving cars making mistakes leading to accidents (e.g., Uber Self-Driving Car Incident).

#### Key Takeaways

- Complementary Roles: Machines enhance human capabilities but do not replace human intelligence.
- Ethical Considerations: Responsible AI development is crucial to mitigate deception and bias.
- Continuous Learning: Both humans and machines must evolve to address emerging challenges.

## Day 2

Experimenting and understanding concepts with the Beispielprojekt_Teil1a.ipynb

### Correlation Analysis

#### What is a correlation?

Correlation measures the strength and direction of the linear relationship between two variables. It indicates how changes in one variable are associated with changes in another.

#### What is the correlation coefficient?

The correlation coefficient is a numerical value that quantifies the degree of correlation between two variables. It ranges from -1 to 1, where:

- **1** indicates a perfect positive correlation,
- **-1** indicates a perfect negative correlation, and
- **0** indicates no correlation.

#### Between which limits does it lie?

The correlation coefficient lies between **-1** and **1**.

#### When is a correlation good?

A correlation is considered strong (good) when the absolute value of the correlation coefficient is close to **1** (e.g., |r| > 0.7). This implies a strong positive or negative linear relationship between the variables.

### Correlation Matrix Analysis

#### Why is the analysis limited to `corr_matrix["median_house_value"]`?

The analysis focuses on `corr_matrix["median_house_value"]` to examine how each feature in the dataset correlates specifically with the target variable (ziel), which is the median house value. This helps identify which features are most influential in predicting house prices.

#### Why is a column missing in `corr_matrix["median_house_value"]`?

The column corresponding to `median_house_value` is missing because a variable does not correlate with itself. In the correlation matrix, the diagonal elements (where a variable would correlate with itself) are typically omitted or set to a default value since they are always perfectly correlated (r = 1).

### Scatter Matrix Visualization

#### Which technique (function and module) was used to create the images in the scatter matrix?

The scatter matrix was created using the `scatter_matrix` function from the `pandas.plotting` module.

        from pandas.plotting import scatter_matrix

        attributes = ["median_house_value", "median_income", "total_rooms",
                    "housing_median_age"]
        scatter_matrix(housing[attributes], figsize=(9, 6))
        save_fig("scatter_matrix_plot")  # extra code
        plt.show()

#### What does the diagonal of the images show?

The diagonal of the scatter matrix displays histograms of each individual variable. These histograms show the distribution of values for each feature.

#### What is a histogram?

A histogram is a graphical representation that organizes a group of data points into user-specified ranges. It shows the frequency distribution of a dataset, illustrating the number of data points that fall within each range.

#### Which of the images (scatter plots) show relationships?

Scatter plots off the diagonal display the relationships between pairs of variables. If the points in a scatter plot show an upward or downward trend, it indicates a positive or negative correlation, respectively. Plots with no discernible pattern suggest little to no correlation.

#### Justify the selection of columns underlying the scatter matrix

The columns selected for the scatter matrix are chosen based on their potential impact on the target variable (`median_house_value`). Features that are likely to influence house prices, such as `median_income`, `total_rooms`, and `housing_median_age`, are included to visualize their relationships with the target and among themselves. They were shosing for having the greatest correlation.

### Observations from Correlation Analysis

#### What is particularly evident here?

It is particularly evident that there is a strong positive correlation between `median_income` and `median_house_value`. This indicates that as the median income in an area increases, the median house value tends to increase as well. Additionally, other features may show varying degrees of correlation, highlighting their influence on house prices.

### Aditional Resources

- [Pandas Scatter Matrix Documentation](https://pandas.pydata.org/docs/reference/api/pandas.plotting.scatter_matrix.html)
- [Understanding Correlation Coefficients](https://statisticsbyjim.com/basics/correlation-coefficient/)
- [Data Visualization with Matplotlib](https://matplotlib.org/stable/gallery/index.html)

## Day 3

Experimenting and understanding concepts with the Beispielprojekt_Teil1b.ipynb

### Understanding concepts in Pandas

Experimenting and understanding concepts with the pandas_erklärungen.ipynb

#### `iloc`

In Pandas, iloc is a powerful indexing and selection tool that allows you to access rows and columns in a DataFrame by their integer positions. The name iloc stands for integer location.

It is primarily used for integer-based indexing to select data by row and column numbers.

                df.iloc[<row_selection>, <column_selection>]

#### `random.permutation`

numpy.random.permutation is a function in NumPy that randomly permutes the elements of an array or generates a permuted sequence of integers.

        Original array: [1 2 3 4 5]
        Shuffled array: [3 1 5 2 4]

#### Stratification in Statistics

Stratification is a sampling technique used in statistics to divide a population into distinct subgroups, known as strata, that are homogeneous in certain characteristics. This method ensures that each subgroup is adequately represented in the overall sample, enhancing the accuracy and reliability of statistical estimates.

1. Identify Strata: Determine the relevant characteristics (e.g., age, gender, income) to divide the population into strata.
2. Divide the Population: Split the entire population into these non-overlapping strata based on the chosen characteristics.
3. Sample Within Strata: Perform random sampling within each stratum. The number of samples from each stratum can be proportional to its size in the population or based on other criteria.

#### `drop`

This line of code removes the column named "income_cat" from the DataFrame set1. Here's a breakdown of what each part does:

set1: The original Pandas DataFrame from which you want to remove a column.
.drop("income_cat", axis=1, inplace=True):
"income_cat": The name of the column you want to delete.
axis=1: Specifies that you're dropping a column. (Use axis=0 to drop rows.)
inplace=True: Modifies the original DataFrame set1 directly without creating a new DataFrame.

#### `from sklearn.impute import SimpleImputer`

The code imports the SimpleImputer, creates an imputer that fills in missing data using the median of each column, and displays the imputer's configuration.

Strategy Options: Besides "median", SimpleImputer can use other strategies like "mean", "most_frequent", or "constant".

Fit and Transform: fit() calculates the required statistics (e.g., median), and transform() applies them to replace missing values.

#### ``fit()`` Function in SimpleImputer

The ``fit()`` function in SimpleImputer is used to compute the necessary statistics for imputing missing values based on the chosen strategy. When you call ``fit()`` on your dataset, the imputer calculates values like the median, mean, or most frequent value for each column, depending on the specified strategy. These computed statistics are then stored within the imputer instance and are essential for accurately replacing the missing data during the transformation phase. Essentially, ``fit()`` learns the parameters needed to handle missing values from the training data.

#### ``transform()`` Function in SimpleImputer

The ``transform()`` function applies the imputation strategy to the dataset using the statistics calculated during the ``fit()`` step. When you invoke ``transform()``, the imputer replaces all identified missing values in the dataset with the corresponding median, mean, most frequent value, or a constant, as defined by the strategy parameter. This function returns a new dataset where the missing values have been filled in, ensuring that the data is complete and ready for further analysis or modeling. In summary, ``transform()`` utilizes the learned statistics to effectively handle and replace missing data points in your dataset.

#### Training and Test separation

Usaually it is devided 80:20 (Pareto) train to test reatio but it deppends on your data set and conditions.

What label mean on this context? A label is the target variable or the outcome that the model is designed to predict. It represents the information you want the model to learn from the data.

Labels are essential for supervised learning tasks as they provide the ground truth the model aims to predict.

During train-test separation, both the training and test sets include labels to facilitate learning and evaluation.

Ensuring that labels are correctly assigned and consistently used across both sets is vital for building effective and reliable machine learning models.

Examples:

- Classification Task: In an email spam filter, the label could be spam or not spam.
- Regression Task: In predicting house prices, the label would be the actual price of the house.

## Day 4

### Continue of machine training

beispielprojekt_teil2.ipynb

### ``OrdinalEncoder`` in scikit-learn

The ``OrdinalEncoder`` from ``sklearn.preprocessing`` is a tool used to convert categorical features into ordinal integers. This is particularly useful for machine learning algorithms that require numerical input. Unlike one-hot encoding, which creates binary columns for each category, ordinal encoding assigns each unique category a distinct integer based on alphabetical ordering or a specified mapping.

#### ``fit()`` Function

The ``fit()`` function in ``OrdinalEncoder`` analyzes the input data to identify all unique categories in each categorical feature. It learns the mapping of each category to an integer based on the specified encoding strategy. During this process, ``fit()`` stores the unique categories and their corresponding integer values within the encoder instance, preparing it for the transformation of the dataset. Essentially, ``fit()`` teaches the encoder how to translate categorical data into numerical form.

#### ``transform()`` Function

The ``transform()`` function applies the learned category-to-integer mappings to the dataset. When you call ``transform()``, the OrdinalEncoder replaces each categorical value in the dataset with its corresponding integer as determined during the fit() step. This results in a numerical representation of the original categorical features, making the data suitable for input into machine learning models. The ``transform()`` function ensures that the categorical data is consistently and accurately encoded based on the previously learned mappings.

#### ``fit_transform()`` Function

The ``fit_transform()`` function is a convenient method that combines the actions of ``fit()`` and ``transform()`` into a single step. When you use ``fit_transform()``, the OrdinalEncoder first learns the category mappings from the data and then immediately applies these mappings to transform the dataset. This is particularly useful for streamlining the preprocessing workflow, allowing you to efficiently encode categorical features without having to call ``fit()`` and ``transform()`` separately.

### `OneHotEncoder` in scikit-learn

The OneHotEncoder from sklearn.preprocessing is a preprocessing tool used to convert categorical features into a binary (0 or 1) representation. Unlike OrdinalEncoder, which assigns a unique integer to each category, OneHotEncoder creates separate binary columns for each unique category within a feature. This encoding method is essential for machine learning algorithms that cannot work with categorical data directly and helps prevent the algorithm from assuming any ordinal relationship between categories.

#### `fit()` Function

The fit() function in OneHotEncoder analyzes the input data to identify all unique categories in each categorical feature. During this process, it determines the necessary binary columns needed to represent each category uniquely. The fit() method stores the unique categories and their corresponding binary encodings within the encoder instance, preparing it for transforming the dataset. Essentially, fit() learns the structure of the categorical data to ensure accurate and consistent encoding during transformation.

#### `transform()` Function

The transform() function applies the learned binary mappings to the dataset. When you call transform(), the OneHotEncoder converts each categorical value into a binary vector where only the corresponding category's column is marked as 1, and all others are 0. This results in a sparse matrix where categorical features are represented in a format suitable for machine learning models. The transform() function ensures that the categorical data is accurately encoded based on the mappings established during the fit() step.

#### `fit_transform()` Function

The fit_transform() function combines the actions of fit() and transform() into a single step. By using fit_transform(), the OneHotEncoder first learns the category mappings from the data and then immediately applies these mappings to transform the dataset. This method is particularly useful for streamlining the preprocessing workflow, allowing for efficient encoding of categorical features without needing to call fit() and transform() separately. It simplifies the code and enhances computational efficiency during the encoding process.

#### Difference Between OneHotEncoder and OrdinalEncoder

OrdinalEncoder assigns a unique integer to each category, introducing an implicit ordinal relationship between categories. In contrast, OneHotEncoder creates separate binary columns for each category, eliminating any implied order. This fundamental difference affects how machine learning models interpret the encoded data.

#### Advantages of OneHotEncoder

- No Ordinal Assumption: Prevents algorithms from assuming a natural order among categories, which is beneficial for nominal data.
- Model Compatibility: Essential for algorithms that require numerical input without ordinal relationships, such as linear regression and neural networks.
- Improved Performance: Can lead to better model performance by providing a clear and distinct representation of categories.

#### Disadvantages of OneHotEncoder

- Increased Dimensionality: Can significantly increase the number of features, especially with high-cardinality categorical variables, leading to the "curse of dimensionality."
- Sparse Representation: Results in a sparse matrix, which can be less efficient in terms of memory and computation for very large datasets.
- Potential Overfitting: More features can sometimes lead to overfitting, especially with limited data.

#### Advantages of OrdinalEncoder

- Simplicity: Easier to implement with fewer resulting features.
- Efficiency: Maintains the original number of features, making it more memory-efficient.
- Suitable for Ordinal Data: Ideal for categorical variables with a meaningful order (e.g., education levels).

#### Disadvantages of OrdinalEncoder

- Implicit Ordinal Relationship: May introduce unintended ordinal relationships where none exist, potentially misleading the model.
- Limited Use Cases: Not suitable for nominal data without inherent order, as it can degrade model performance.

### Feature Scaling in Machine Learning (Skalierung)

Feature scaling is a crucial preprocessing step in machine learning that involves transforming numerical features to a common scale without distorting differences in the ranges of values. This ensures that all features contribute equally to the model's performance, especially for algorithms sensitive to the scale of data, such as gradient descent-based methods and distance-based algorithms.

#### Min-Max Scaling (`MinMaxScaler`)

MinMaxScaler transforms features by scaling each feature to a given range, typically between 0 and 1. It preserves the relationships among the original data points by maintaining the distribution shape but compresses the range to the specified scale.

Advantages:

- Preserves the relationships and distribution of the data.
- Simple and effective for scaling features to a specific range.

Disadvantages:

- Sensitive to outliers, which can skew the scaling parameters.

#### Standardization (`StandardScaler`)

StandardScaler standardizes features by removing the mean and scaling to unit variance. This transformation results in a distribution with a mean of 0 and a standard deviation of 1, making it suitable for algorithms that assume normally distributed data.

Advantages:

- Not bounded to a specific range, preserving useful information about the distribution.
Less affected by outliers compared to Min-Max Scaling.

Disadvantages:

- Assumes that the data follows a Gaussian distribution, which may not always be the case.

#### Normalization (Normalizer)

Normalizer scales individual samples to have unit norm (e.g., L1 or L2 norm). This is particularly useful for text classification or clustering tasks where the direction of the data points matters more than their magnitude.

Advantages:

Ensures that each sample contributes equally to the analysis, regardless of their original magnitude.

- Useful for algorithms that rely on distance calculations, such as k-nearest neighbors.

Disadvantages:

- Does not account for the distribution of features across the dataset.
- Not suitable for all types of data, especially where feature scaling based on global statistics is needed.

#### RobustScaler

The RobustScaler from sklearn.preprocessing is a feature scaling tool designed to mitigate the impact of outliers in your dataset. Unlike other scalers that use mean and standard deviation, RobustScaler utilizes the median and the Interquartile Range (IQR) to scale features. This makes it particularly effective for datasets with significant outliers, ensuring that the scaling process is not unduly influenced by extreme values.

Advantages:

- Outlier Resistance: By using median and IQR, RobustScaler is less sensitive to outliers compared to MinMaxScaler and StandardScaler.
- Effective Scaling: Ensures that the majority of the data points are scaled within a similar range, improving model performance.
- Preserves Data Distribution: Maintains the shape of the original distribution without being skewed by extreme values.

Disadvantages:

- Not Ideal for All Data: If your dataset does not contain outliers, other scalers like StandardScaler might be more appropriate.
- Less Intuitive Interpretation: Scaling based on median and IQR may be less straightforward to interpret compared to mean-based scaling.

#### Comparison and Choosing the Right Scaler

Min-Max Scaling is ideal when you need to bound your data within a specific range and when the algorithm is sensitive to the scale of the data.

Standardization is preferred when the data follows a Gaussian distribution and when algorithms assume normally distributed data.

Normalization is best suited for scenarios where the magnitude of individual samples is important, such as in text processing or when using distance-based algorithms.

Be mindful of outliers, as they can significantly impact scaling, especially with Min-Max Scaling.

It’s often beneficial to experiment with different scaling techniques to determine which one yields the best model performance.

### FunctionTransformer

The FunctionTransformer from sklearn.preprocessing is a versatile tool that allows you to apply custom functions to your data as part of a preprocessing pipeline. It enables the integration of any user-defined transformation function, making it easy to include bespoke data manipulation steps alongside standard preprocessing techniques. This flexibility is particularly useful when you need to perform unique transformations that are not covered by existing transformers in scikit-learn.

- `fit()`: Typically does nothing and returns the transformer instance, maintaining interface consistency.
- `transform()`: Applies the user-defined function to transform the data.
- `fit_transform()`: Combines fitting and transforming, effectively performing the transformation in one step.

### Understanding `np.c_[X_neu, neue_spalte]`

The line of code X_neu = np.c_[X_neu, neue_spalte] uses NumPy's c_ object to concatenate arrays column-wise. Here’s what each part does:

np.c_: A NumPy utility that facilitates column-wise stacking of arrays.
X_neu: An existing NumPy array or matrix.
neue_spalte: A new column (1D array) that you want to add to X_neu.
Functionality:

Concatenation: Combines X_neu and neue_spalte horizontally, adding neue_spalte as the last column of X_neu.
Result: The variable X_neu is updated to include the new column, effectively expanding its dimensionality.

## Day 5

### Data Preprocessing Pipeline Overview

In machine learning workflows, data typically undergoes a series of preprocessing steps to prepare it for modeling. A common pipeline includes:

Imputer (Imputer()): Handles missing values in the dataset by replacing them with statistical measures like mean, median, or mode.
Encoder (Encoder()): Transforms categorical variables into numerical formats using techniques such as One-Hot Encoding or Ordinal Encoding.
Scaler (Scaler()): Standardizes or normalizes numerical features to ensure they contribute equally to the model's performance.
Predictions (Voraussagen): The processed data is then used to train machine learning models to make predictions.
Evaluation (Bewerten): Assesses the model's performance using metrics like accuracy, precision, recall, or RMSE.

X -> Imputer() -> Encoder() -> Scaler() -> Voraussagen -> Bewerten

Scaler:
MinMax --> [0,1]\
Standard --> meistens [-1,1]
Robust --> Nicht-Ausreißer[-1,+1]
Maxabs --> [-1,+1]

Encoder:
OneHot --> viele neue Spalten
Ordinal --> 1 neue Spalte, Stufenfolge

### Training and Evaluation of Machine Learning Models

In machine learning, training involves teaching a model to recognize patterns in data using the ``fit`` method, while prediction uses the learned patterns to make forecasts on new data with the ``predict`` method. Evaluation assesses the model's performance using metrics such as Mean Squared Error (MSE) or R². Understanding how different models operate and their strengths and weaknesses is crucial for selecting the right algorithm for your task.

#### Linear Regression

Linear Regression models the relationship between a dependent variable and one or more independent variables by fitting a linear equation to the observed data.

- fit(): Calculates the best-fitting line by minimizing the sum of squared differences between observed and predicted values.
- predict(): Uses the learned coefficients to estimate the dependent variable for new data points.

Advantages:

- Simple and easy to interpret.
- Computationally efficient.
- Works well when there is a linear relationship between features and the target.

Disadvantages:

- Assumes a linear relationship, which may not hold true for complex data.
- Sensitive to outliers.
- Can underperform when features are highly correlated (multicollinearity).

#### Decision Tree Regressor

Decision Tree Regressors split the data into subsets based on feature values, creating a tree-like model of decisions to predict the target variable.

- fit(): Recursively splits the dataset based on feature values that result in the highest information gain or lowest variance.
- predict(): Traverses the tree based on feature values of new data to arrive at a prediction.

Advantages:

- Handles both numerical and categorical data.
- Captures non-linear relationships.
- Easy to visualize and interpret.

Disadvantages:

- Prone to overfitting, especially with deep trees.
- Can be unstable; small changes in data may lead to different trees.
- Often requires pruning to improve generalization.

#### Random Forest Regressor

Random Forest Regressors are ensemble models that build multiple decision trees and aggregate their predictions to improve accuracy and control overfitting.

- fit(): Trains multiple decision trees on random subsets of the data and features.
- predict(): Averages the predictions from all individual trees to produce the final output.

Advantages:

- Reduces overfitting compared to individual decision trees.
- Handles large datasets with higher dimensionality.
- Provides feature importance insights.

Disadvantages:

- More complex and computationally intensive.
- Less interpretable than single decision trees.
- Can be slower to predict due to the ensemble of trees.

#### Support Vector Machine (SVM) Regressor

Support Vector Machines for regression (SVR) aim to find a function that deviates from the actual observed targets by a value no greater than a specified margin.

- fit(): Determines the optimal hyperplane that fits within the margin of tolerance for all data points.
- predict(): Uses the learned hyperplane to make predictions on new data.

Advantages:

- Effective in high-dimensional spaces.
- Uses kernel trick to model non-linear relationships.
- Robust to outliers with appropriate kernel choice.

Disadvantages:

- Requires careful tuning of parameters and kernel selection.
- Computationally intensive for large datasets.
- Less interpretable compared to linear models and decision trees.

### Summary of Model Training

- Training (fit): The process where the model learns from the training data by adjusting its parameters to minimize error.
- Prediction (predict): Using the trained model to make predictions on new, unseen data.
- Evaluation: Assessing model performance using metrics like MSE, R², MAE, etc., to determine how well the model generalizes to new data.
