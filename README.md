# Credit Card Fraud Detection: Dimensionality Reduction and Classification

**Author:** Himmler Benitez
**Date:** December 2025

---

## 1. Objective
The main objective of this analysis is to apply dimensionality reduction techniques to a highly imbalanced credit card transaction dataset to uncover latent behavioral patterns that improve downstream fraud detection performance. The project focuses on learning compact representations that preserve meaningful transaction structures while reducing noise and redundancy.

**Business Value:**
* Improve model robustness in the presence of class imbalances.
* Reduce computational complexity while enabling a more interpretable classification model.
* Provide an insightful vision of the risk associated with each transaction.

---

## 2. Dataset Description
The analysis utilizes the Credit Card Fraud Detection dataset obtained from Kaggle.
* **Content:** Simulated credit card transactions labeled as either legitimate or fraudulent.
* **Extreme Imbalance:** Contains only 151 fraudulent samples in contrast to 9,849 legitimate samples.
* **Context:** This represents a realistic and challenging fraud detection scenario.

---

## 3. Data Exploration & Preparation

### Data Exploration
* **Skewness:** High right-skewness observed in transaction amount and velocity.
* **Distributions:** Near-uniform distributions for contextual and demographic features.
* **Correlation Analysis:** Most feature pairs exhibit very low linear correlation ($|\rho|\approx0.00-0.10$), indicating that fraudulent behavior is governed by subtle, multivariate patterns.



### Preparation Pipeline
A technical pipeline was developed to ensure clean data preparation and prevent leakage:
* **Standardization:** Normalized numerical features to a standard scale.
* **Log-Transformation:** Applied to skewed variables to stabilize variance.
* **SMOTE:** Utilized to create synthetic data for the minority class to compensate for underrepresentation.
* **Stratified Splitting:** Preserved fraud ratios across training and testing sets.

---

## 4. Unsupervised Model Variations
Several unsupervised methods were explored to learn latent transaction structures:
* **PCA (95% Explained Variance):** Linear dimensionality reduction used as a foundation for downstream modeling.
* **K-Means Clustering:** Applied to PCA-reduced data to explore latent groupings.
* **Kernel PCA:** Evaluated to capture non-linear behavioral patterns.

---

## 5. Classifier Model & Implementation
A `RandomizedSearchCV` approach was executed to optimize hyperparameters, specifically comparing linear PCA (90% and 95% variance) with Kernel PCA using an RBF kernel.

```python
# Pipeline configuration for Logistic Regression
def get_pipeline(clf):
    pipeline = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("pca", PCA()),
        ("smote", SMOTE(random_state=42)),
        ("clf", clf)
    ])
    return pipeline

# Parameters for Random Search comparison
param_distributions = {
    "pca": [PCA(random_state=42), KernelPCA(kernel="rbf")],
    "pca__n_components": [0.90, 0.95, 10, 15, 20],
    "clf__C": loguniform(1e-2, 1e1),
    "clf__class_weight": [None, "balanced"]
} 
```

## 6. Results and Performance
The Logistic Regression model, trained on PCA-reduced and SMOTE-balanced data, demonstrated strong discriminative capability.
Metric,Class 0 (Legit),Class 1 (Fraud),Overall
Precision,1.00,0.31,97% Accuracy
Recall,0.97,0.93,0.9934 ROC-AUC
F1-Score,0.98,0.47,0.98 Weighted Avg



## 7. Conclusion
The findings indicate that Kernel PCA consistently outperformed linear PCA, suggesting that fraudulent behavior is driven by non-linear patterns not captured by linear projections. Unsupervised dimensionality reduction proved to be an effective tool for fraud detection in highly imbalanced settings.

### Recommendations:

* Combine Kernel PCA with interpretable classifiers to balance performance and deployment scalability.

### Future Work: Implement a 1D CNN or SVM model to potentially further improve performance for this data type.
