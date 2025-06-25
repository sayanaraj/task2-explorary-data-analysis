# Task 2 – Exploratory Data Analysis (EDA) on Titanic Dataset

## Objective

The objective of this task is to perform Exploratory Data Analysis (EDA) on the Titanic dataset using Python. This includes visualizing data distributions, detecting patterns and anomalies, identifying feature relationships, and generating statistical summaries to better understand the data and prepare it for machine learning applications.

---

## Dataset Information

- **Dataset Name**: Titanic Dataset
- **Source**: [Kaggle - Titanic Dataset](https://www.kaggle.com/datasets/yasserh/titanic-dataset)
- **File Used**: titanic.csv

The dataset contains demographic and survival information for passengers aboard the Titanic. The primary feature of interest is `Survived`, which indicates whether a passenger survived (1) or not (0).

---

## Libraries and Tools

- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Seaborn

---

## EDA Process Overview

### 1. Data Loading and Preliminary Inspection
- The dataset was loaded using the Pandas library.
- Basic structure and properties of the data were explored using functions such as `.head()`, `.info()`, and `.describe()`.
- Missing values were identified using `.isnull().sum()`.

### 2. Handling Missing Data
- Missing values in the `Age` column were imputed using the median to reduce the influence of outliers.
- Missing values in the `Embarked` column were filled using the mode (most frequent value).
- The `Cabin` column was removed due to a high proportion of missing entries.

### 3. Descriptive Statistics
- Summary statistics for numeric variables such as `Age` and `Fare` were generated using `.describe()`.
- Frequency counts for categorical variables like `Sex`, `Embarked`, and `Survived` were computed using `.value_counts()`.

### 4. Data Visualization

#### Histograms
- Histograms for `Age` and `Fare` were created to examine the distribution and skewness of numeric variables.

#### Boxplots
- Boxplots were used to compare `Age` and `Fare` across different values of `Survived`, and to examine variability across passenger classes (`Pclass`).

#### Countplots
- Countplots were created to analyze the distribution of `Survived` with respect to categorical variables such as `Sex`, `Pclass`, and `Embarked`.

#### Correlation Matrix and Heatmap
- A correlation matrix was generated to study the linear relationships between numeric features.
- A heatmap was used to visualize correlation coefficients with clear annotation.

#### Pairplot
- A pairplot was created to visualize bivariate relationships between `Age`, `Fare`, `Pclass`, and `Survived`, and to identify trends or separable clusters.

---

## Key Findings

- Female passengers had a significantly higher survival rate compared to male passengers.
- Passengers in the first class had the highest survival rate, while those in the third class had the lowest.
- Younger passengers were more likely to survive than older passengers.
- Higher ticket fares were generally associated with a greater likelihood of survival.
- A negative correlation was observed between `Pclass` and `Fare`, indicating that higher classes paid more.

