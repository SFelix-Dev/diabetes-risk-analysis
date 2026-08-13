# Diabetes & Cardiovascular Risk Analysis

End-to-end data science project analyzing the CDC BRFSS 2015 Diabetes Health Indicators dataset. The project covers exploratory data analysis, hypothesis testing, classification modeling, and unsupervised learning across four parts.

## Dataset

**CDC Behavioral Risk Factor Surveillance System (BRFSS) 2015**
- 253,680 survey responses, 21 health indicator features
- Target variable: `Diabetes_012` (0 = No diabetes, 1 = Pre-diabetic, 2 = Diabetic)
- Source: [UCI Machine Learning Repository](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)

## Project Structure

```
diabetes-risk-analysis/
├── ProjectP1_FelixShyann_Z23813474.ipynb   # EDA
├── ProjectP2_FelixShyann_Z23813474.ipynb   # Hypothesis Testing
├── ProjectP3_FelixShyann_Z23813474.ipynb   # Classification
├── ProjectP4_FelixShyann_Z23813474.ipynb   # PCA & Clustering
├── diabetes_012_health_indicators_BRFSS2015.csv
├── diabetes_binary_health_indicators_BRFSS2015.csv
└── diabetes_binary_5050split_health_indicators_BRFSS2015.csv
```

## Part 1 — Exploratory Data Analysis

- Removed 23,899 duplicate rows from the dataset
- Engineered a `HealthyDiet` feature from fruit and vegetable consumption variables
- Computed descriptive statistics for BMI and physical health across diabetes status
- **Visualizations:** BMI histogram, Age distribution, BMI vs. High Blood Pressure box plot, Physical Health vs. Diabetes Status box plot, Heart Disease Rate bar chart, High Cholesterol Rate bar chart, Physical Health KDE density plot

## Part 2 — Hypothesis Testing

**T-Tests (scipy)**
- Mental health days vs. high blood pressure status → significant difference (p < 0.05)
- Income vs. heart disease history → significant difference (p < 0.05)

**Chi-Square Tests**
- Stroke history vs. high cholesterol → significant association (p ≈ 0)
- Sex vs. heavy alcohol consumption → significant association (p = 0.004)

**Linear Regression (pingouin)**
- Education → Income (controlling for Age, Sex): R² = 0.22, positive relationship
- Heavy alcohol consumption → BMI (controlling for PhysActivity, Age): R² = 0.027, negative relationship

## Part 3 — Classification

**Prediction task:** Classify diabetes status (0/1/2) using 6 predictors including one-hot encoded Age groups

**Models compared via 5-fold cross-validation:**
- Logistic Regression
- Decision Tree (max depth = 10)
- Random Forest (100 estimators) ← best performer

**Techniques:**
- Handled class imbalance using `RandomOverSampler`
- Evaluated with accuracy, precision, recall, and F1 score
- Compared L1 (Lasso), L2 (Ridge), and Elastic Net regularization
- L1/L2 achieved ~63% accuracy; Elastic Net zeroed out Smoker, PhysActivity, and HighBP for pre-diabetic classification

## Part 4 — PCA & K-Means Clustering

**PCA:**
- Standardized features using `StandardScaler`
- Fit PCA across all components; selected 14 components to reach 90% explained variance
- Visualized top 3 principal components in a 3D scatter plot

**K-Means Clustering:**
- Used elbow method to determine optimal k = 3
- Ran K-Means on all 21 dataset features
- Analyzed cluster centers to profile groups by BMI, blood pressure, cholesterol, physical activity, and heart disease risk

## Technologies

- Python, Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn (LogisticRegression, DecisionTree, RandomForest, PCA, KMeans, StandardScaler)
- imbalanced-learn (RandomOverSampler)
- SciPy, Pingouin

## How to Run

1. Clone the repo
2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn scipy pingouin
   ```
3. Open any notebook in Jupyter and run all cells in order
