# Titanic Data Science Project: Classification & Regression

This repository features an end-to-end data science pipeline using the Titanic dataset[cite: 1]. The project covers data cleaning, exploratory data analysis (EDA), feature engineering, and the implementation of robust machine learning workflows for both classification (predicting survival) and regression (predicting ticket fare)[cite: 1].

---

## 🛠️ Installation & Requirements

To run this project, make sure you have the following Python libraries installed[cite: 1]:
* `pandas`[cite: 1]
* `numpy`[cite: 1]
* `matplotlib`[cite: 1]
* `seaborn`[cite: 1]
* `scikit-learn`[cite: 1]

---

## 📂 Project Workflow

### 1. Data Cleaning & Preprocessing
* **Missing Value Imputation:** Handled missing data by filling numerical missing values (`age`) with the median[cite: 1]. Categorical variables like `boat` and `home.dest` were filled with a string placeholder or mode[cite: 1]. 
* **Row Removal:** Dropped rows where critical target/feature information (`fare`, `embarked`) was missing[cite: 1].
* **Dropping Features:** Removed non-predictive or heavily empty columns (`body`, `cabin`) to prevent model noise[cite: 1].
* **Outlier Detection:** Used Seaborn boxplots to isolate and explore high-value outliers within passenger fares[cite: 1].

### 2. Feature Engineering & Encoding
* **Categorical Encoding:** Applied `LabelEncoder` for binary text data (`sex`) and implemented One-Hot Encoding (`pd.get_dummies`) for geographical origins (`embarked`)[cite: 1].
* **Family Size Metric:** Aggregated `sibsp` and `parch` columns to create a new `family_size` feature[cite: 1].
* **Travel Status:** Built a binary indicator feature (`travelled_alone`) based on family configurations to measure impact on survival rates[cite: 1].

---

## 📊 Exploratory Data Analysis & Visualizations

The project generates several key analytical charts to uncover underlying trends[cite: 1]:
* **Survival Rate by Family Size:** A bar chart exploring how traveling with relative groups affected survival probabilities[cite: 1].
* **Survival Rate Based on Traveling Alone:** Examining the correlation between independence and safety outcomes[cite: 1].
* **Feature Correlation Matrix:** A complete annotated Seaborn heatmap showing structural linear relationships across all processed numeric elements[cite: 1].

---

## 🤖 Modeling Frameworks

The core modeling architecture leverages `scikit-learn` Pipelines combined with `ColumnTransformer` layouts[cite: 1]. Numerical columns are scaled via `StandardScaler`, and categorical text elements are dynamically processed through a `OneHotEncoder` sub-pipeline[cite: 1].

### Part 1: Classification Problem (Target: `survived`)
We tested multiple classification structures optimized via 5-Fold Stratified `GridSearchCV`[cite: 1]:

| Model | Hyperparameters Checked | Cross-Validation Accuracy Score |
| :--- | :--- | :--- |
| **Logistic Regression** | `C: [0.01, 0.1, 1, 10, 100]`, `penalty: ['l2']` | **97.61%** (Best Choice)[cite: 1] |
| **Support Vector Classifier (SVC)** | `C: [0.1, 1, 10]`, `kernel: ['rbf', 'linear', 'poly']` | **97.61%** (Best Choice)[cite: 1] |
| **Random Forest Classifier** | `n_estimators`, `max_depth`, `min_samples_split` | **97.61%** (Best Choice)[cite: 1] |
| **K-Nearest Neighbors (KNN)** | `n_neighbors: [3, 5, 7, 9, 11, 13]`, `weights` | 93.10%[cite: 1] |

### Part 2: Regression Problem (Target: `fare`)
The feature set was reshuffled to forecast ticket prices using negative mean squared error (`neg_mean_squared_error`) evaluation scoring metrics[cite: 1]:
* **Linear Regression:** Baseline continuous target evaluation[cite: 1].
* **Support Vector Regressor (SVR):** Evaluated over multiple configurations (`rbf`, `linear`)[cite: 1].
* **Random Forest Regressor:** Built with ensemble parameter combinations adjusting tree-depth and splitting boundaries[cite: 1].

---

## 🚀 How to Run the Project
1. Clone this repository or open the notebook file directly in **Google Colab**[cite: 1].
2. Set up your notebook connection paths to load the raw `Titanic.csv` data folder[cite: 1].
3. Run all initialization and pipeline grid cells to re-compile model parameters and plots[cite: 1].
