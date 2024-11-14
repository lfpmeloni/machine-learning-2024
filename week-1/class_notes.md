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
