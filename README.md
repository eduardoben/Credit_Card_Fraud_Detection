# Credit Card Fraud Detection: Dimensionality Reduction and Classification

**Author:** Himmler Benitez  
[cite_start]**Date:** December 2025 [cite: 5]

## 1. Objective
[cite_start]The main objective of this analysis is to apply dimensionality reduction techniques to a highly imbalanced credit card transaction dataset to uncover latent behavioral patterns that improve downstream fraud detection performance[cite: 9]. [cite_start]The project focuses on learning compact representations that preserve meaningful transaction structure while reducing noise and redundancy[cite: 10]. 

The business value of this project is to:
* [cite_start]Improve model robustness in the presence of class imbalances[cite: 11].
* [cite_start]Reduce computational complexity while enabling a more interpretable classification model[cite: 11].
* [cite_start]Provide an insightful vision of the risk associated with each transaction[cite: 11].

---

## 2. Dataset Description
[cite_start]The analysis uses the Credit Card Fraud Detection dataset obtained from Kaggle[cite: 13].
* [cite_start]**Content:** Simulated credit card transactions labeled as legitimate or fraudulent[cite: 14].
* [cite_start]**Imbalance:** The dataset is extremely unbalanced, containing only 151 fraudulent samples compared to 9,849 legitimate samples[cite: 15].
* [cite_start]**Challenge:** This represents a realistic and challenging fraud detection scenario[cite: 16].

---

## 3. Data Exploration & Preparation

### Exploration Highlights
* [cite_start]**Skewness:** Highly right-skewed distributions were observed for transaction amount and velocity[cite: 21].
* [cite_start]**Distribution:** Contextual and demographic features showed near-uniform distributions[cite: 22].
* [cite_start]**Correlation:** Analysis showed no dominant linear drivers of fraud ($|\rho|\approx0.00-0.10$), indicating that fraudulent behavior is governed by subtle, multivariate patterns[cite: 194, 196].

### Preparation Pipeline
[cite_start]A technical pipeline was created to prepare the data for modeling[cite: 201]:
1. [cite_start]**Standardization:** Normalized the scale of numerical features[cite: 202].
2. [cite_start]**Log-Transformation:** Stabilized variance for skewed variables[cite: 203].
3. [cite_start]**SMOTE:** Created synthetic data to compensate for the minority class representation in the training split[cite: 205].
4. [cite_start]**Stratified Splitting:** Preserved fraud ratios during the train-test split[cite: 204].



---

## 4. Unsupervised Model Variations
[cite_start]The project explored several unsupervised techniques to learn latent structures[cite: 206]:
* [cite_start]**PCA (95% Explained Variance):** Linear dimensionality reduction to reduce the feature space[cite: 207, 209].
* [cite_start]**Kernel PCA:** Used to capture non-linear patterns through an RBF kernel[cite: 305].
* [cite_start]**K-Means Clustering:** Applied to PCA-reduced data to explore latent groupings of transactions[cite: 211, 212].

---

## 5. Implementation
[cite_start]The project utilized `RandomizedSearchCV` with a stratified 3-fold cross-validation to optimize the pipeline and hyperparameters[cite: 243, 246, 247].

```python
# Example of the pipeline configuration used
def get_pipeline(clf):
    pipeline = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("pca", PCA()),
        ("smote", SMOTE(random_state=42)),
        ("clf", clf)
    ])
    return pipeline
