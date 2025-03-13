## Details
**Full Name: Ezekiel Ngungu Kathoka**

**School: Moringa School**

**Session: Hybrid**

**Phase: Phase 3 Project**

# Water Well Condition Prediction in Tanzania
## Overview
This project aims to predict the condition of water wells in Tanzania to assist stakeholders—such as NGOs, government bodies, and water management organizations—in optimizing resource allocation, prioritizing maintenance efforts, and planning future well construction. The goal is to classify water wells into three categories:

**Functional**: The water point is fully operational with no repairs needed.

**Functional but need repair**: The water point is working but requires maintenance.

**Non-functional**: The water point is not operational.

By identifying wells that need repair or are non-functional, this project helps ensure reliable access to clean drinking water for communities in Tanzania.

## Problem Statement
Tanzania, a developing country with over 57 million people, faces significant challenges in providing clean and accessible water. Many water wells are installed across the country, but some are in need of repair or have completely failed. This project builds a classification model to predict the condition of water wells, enabling proactive maintenance and efficient resource allocation.

## Objectives
**Proactive Maintenance & Resource Allocation**: Prioritize repairs for high-risk and functional-but-vulnerable wells.

**Identify Key Factors Influencing Water Pump Failures**: Understand the factors contributing to well failures.

**Strategic Planning for New Wells**: Support stakeholders in planning new well installations.

**Support Government & Stakeholders in Water Crisis Management**: Provide actionable insights to improve water access.

## Dataset
The dataset contains information about water wells in Tanzania, including features such as:

Geographical data: Latitude, longitude, region, basin, etc.

Well characteristics: Waterpoint type, extraction type, construction year, etc.

Management data: Scheme management, payment type, installer, etc.

Target variable: status_group (Functional, Functional but needs repair, Non-functional).

## Dataset Details
**Training Data**: training_set_values.csv (features) and training_set_labels.csv (target).

**Test Data**: test_set_values.csv (features for prediction).

## Methodology
### 1. Data Preprocessing
**Handling Missing Values**: Forward fill (ffill) was used to impute missing values.

**Dropping Unnecessary Columns**: Columns like quantity_group, source_type, num_private, and waterpoint_type were dropped.

**Encoding Categorical Variables**: Label encoding was applied to convert categorical variables into numerical format.

**Feature Scaling**: Standard scaling was applied to normalize numerical features.

### 2. Exploratory Data Analysis (EDA)
**Univariate Analysis**: Analyzed the distribution of categorical and numerical features.

**Bivariate Analysis**: Explored relationships between features and the target variable.

**Multivariate Analysis**: Visualized correlations between numerical features using a heatmap.

### 3. Model Training
Six classification models were trained and evaluated:

Logistic Regression

Decision Tree

Random Forest

K-Nearest Neighbors (K-NN)

Support Vector Machine (SVM)

Naïve Bayes

### 4. Model Evaluation
Models were evaluated based on:

Accuracy

Precision, Recall, and F1-Score (macro and weighted averages)

Confusion Matrix

### 5. Hyperparameter Tuning
**Random Forest**: Tuned n_estimators and max_depth.

**Decision Tree**: Tuned max_depth and min_samples_split.

**Naïve Bayes**: Tuned var_smoothing.

## Results
### Model Performance
Model	Accuracy	Precision (Macro Avg)	Recall (Macro Avg)	F1-Score (Macro Avg)
Logistic Regression	0.6382	0.47	0.44	0.43
Decision Tree	0.7379	0.62	0.63	0.62
Random Forest	0.8114	0.74	0.67	0.69
K-NN	0.7468	0.66	0.60	0.62
SVM	0.7562	0.71	0.56	0.57
Naïve Bayes	0.5859	0.48	0.50	0.49

## Key Insights
Random Forest performed the best, achieving the highest accuracy (0.8114) and F1-score (0.69).

Class Imbalance: The minority class (Functional but needs repair) had poor recall across all models, indicating difficulty in identifying this class.

Decision Tree and K-NN were decent alternatives, with accuracy > 0.73.

Naïve Bayes underperformed, likely due to its assumptions not fitting the data well.

## Recommendations
Best Model: Use Random Forest for its superior performance.

Improving Class 1 Performance:

Address class imbalance using techniques like oversampling (e.g., SMOTE) or class weighting.

Experiment with hyperparameter tuning to improve recall for the minority class.

### Alternative Models:

Use Decision Tree or Logistic Regression if interpretability is important.

Use K-NN or SVM if computational efficiency is a concern.

Feature Engineering: Analyze feature importance and consider creating new features to improve model performance.

### Deployment: Deploy the best model for real-world predictions and monitor its performance over time.

## Next Steps
**Advanced Techniques**: Experiment with ensemble methods (e.g., Gradient Boosting, XGBoost) or deep learning models.

**Feature Importance**: Analyze which features contribute most to the model's predictions.

**Stakeholder Collaboration**: Work with NGOs and government bodies to implement the model's recommendations.

## Dependencies
**Python Libraries**: pandas, numpy, matplotlib, seaborn, scikit-learn.

**Dataset**: training_set_values.csv, training_set_labels.csv, test_set_values.csv.

