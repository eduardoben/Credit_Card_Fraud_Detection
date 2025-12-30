# Credit Card Fraud Detection: Dimensionality Reduction and Classification

**Author:** Himmler Benitez  
**Date:** December 2025 [cite: 5]

## 1. Objective
The main objective of this analysis is to apply dimensionality reduction techniques to a highly imbalanced credit card transaction dataset to uncover latent behavioral patterns that improve downstream fraud detection performance[cite: 9]. [cite_start]The project focuses on learning compact representations that preserve meaningful transaction structure while reducing noise and redundancy[cite: 10]. 

The business value of this project is to:
* Improve model robustness in the presence of class imbalances[cite: 11].
* Reduce computational complexity while enabling a more interpretable classification model[cite: 11].
* Provide an insightful vision of the risk associated with each transaction[cite: 11].

---

## 2. Dataset Description
The analysis uses the Credit Card Fraud Detection dataset obtained from Kaggle[cite: 13].
* **Content:** Simulated credit card transactions labeled as legitimate or fraudulent[cite: 14].
* **Imbalance:** The dataset is extremely unbalanced, containing only 151 fraudulent samples compared to 9,849 legitimate samples[cite: 15].
* **Challenge:** This represents a realistic and challenging fraud detection scenario[cite: 16].

---

## 3. Data Exploration & Preparation

### Exploration Highlights
* **Skewness:** Highly right-skewed distributions were observed for transaction amount and velocity[cite: 21].
* **Distribution:** Contextual and demographic features showed near-uniform distributions[cite: 22].
* **Correlation:** Analysis showed no dominant linear drivers of fraud ($|\rho|\approx0.00-0.10$), indicating that fraudulent behavior is governed by subtle, multivariate patterns[cite: 194, 196].

### Preparation Pipeline
A technical pipeline was created to prepare the data for modeling[cite: 201]:
1. **Standardization:** Normalized the scale of numerical features[cite: 202].
2. Log-Transformation:** Stabilized variance for skewed variables[cite: 203].
3. **SMOTE:** Created synthetic data to compensate for the minority class representation in the training split[cite: 205].
4. **Stratified Splitting:** Preserved fraud ratios during the train-test split[cite: 204].



---

## 4. Unsupervised Model Variations
The project explored several unsupervised techniques to learn latent structures[cite: 206]:
* **PCA (95% Explained Variance):** Linear dimensionality reduction to reduce the feature space[cite: 207, 209].
* **Kernel PCA:** Used to capture non-linear patterns through an RBF kernel[cite: 305].
* **K-Means Clustering:** Applied to PCA-reduced data to explore latent groupings of transactions[cite: 211, 212].

---

## 5. Implementation
The project utilized `RandomizedSearchCV` with a stratified 3-fold cross-validation to optimize the pipeline and hyperparameters[cite: 243, 246, 247].

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
