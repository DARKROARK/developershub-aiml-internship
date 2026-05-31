# ❤️ Task 3: Heart Disease Prediction

**DevelopersHub Corporation — AI/ML Engineering Internship**

---

## 📌 Task Objective

Build a binary classification model to predict whether a patient is at risk of heart disease based on 13 clinical features. Evaluate using accuracy, ROC-AUC, and confusion matrix, and identify the most important predictive features.

---

## 📂 Dataset Used

| Property | Detail |
|---|---|
| **Name** | Heart Disease UCI Dataset |
| **Source** | UCI ML Repository via `ucimlrepo` (id=45) / Kaggle |
| **Rows** | 303 patients |
| **Features** | 13 clinical attributes |
| **Target** | Binary — 1 = Heart disease present, 0 = No disease |
| **Disease Rate** | ~54% positive class |

---

## 🛠️ Libraries Used

- `pandas`, `numpy` — data handling
- `scikit-learn` — model training, evaluation, preprocessing
- `matplotlib`, `seaborn` — visualization
- `ucimlrepo` — dataset loading (no Kaggle login needed)

---

## ⚙️ Preprocessing Steps

1. Missing value check — dataset is clean, no imputation needed
2. One-hot encoding for `cp`, `thal`, `slope` (multi-class categoricals)
3. `StandardScaler` applied for Logistic Regression
4. 80/20 stratified train/test split

---

## 🤖 Models Applied

### 1. Logistic Regression
- Linear classifier, outputs probabilities
- Features scaled with `StandardScaler`
- Best for: interpretability, clinical deployment

### 2. Decision Tree Classifier
- `max_depth=5`, `min_samples_leaf=5`
- No scaling needed
- Best for: understanding decision rules, feature importance

---

## 📊 Visualizations Created

| Plot | File |
|---|---|
| Class distribution (bar + pie) | `class_distribution.png` |
| Feature distributions by target | `feature_distributions.png` |
| Categorical feature analysis | `categorical_analysis.png` |
| Correlation heatmap | `correlation_heatmap.png` |
| Confusion matrices (both models) | `confusion_matrices.png` |
| ROC curves with AUC | `roc_curves.png` |
| Feature importances (DT) | `feature_importance.png` |
| LR coefficients | `lr_coefficients.png` |
| Decision tree structure | `decision_tree.png` |

---

## 🔍 Key Results and Findings

1. **Exercise-related features dominate**: `thalach` (max heart rate), `exang` (exercise angina), and `oldpeak` (ST depression) are the strongest predictors.
2. **`ca` (major vessels)** is highly predictive — more blocked vessels = higher risk.
3. **Chest pain type** matters more than cholesterol, contrary to popular assumption.
4. **Cholesterol is a weak predictor** in this dataset — consistent with some cardiology research.
5. Both models achieved solid ROC-AUC, with Logistic Regression preferred for clinical use due to interpretability.

---

## 🚀 How to Run

```bash
# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn ucimlrepo notebook

# Launch notebook
jupyter notebook task3_heart_disease.ipynb
```

> If `ucimlrepo` fails (network issues), place `heart.csv` in the same folder. The notebook auto-falls back to CSV loading.

---

## 📁 File Structure

```
task3/
├── task3_heart_disease.ipynb    # Main Jupyter Notebook
├── heart.csv                    # Fallback dataset
├── class_distribution.png
├── feature_distributions.png
├── categorical_analysis.png
├── correlation_heatmap.png
├── confusion_matrices.png
├── roc_curves.png
├── feature_importance.png
├── lr_coefficients.png
├── decision_tree.png
└── README.md
```

---

*Submitted as part of the DevelopersHub Corporation AI/ML Engineering Internship — Task 3*
