# ML-Practicals
# Machine Learning Practicals

This repository contains Machine Learning practical programs implemented using Python and Scikit-learn.

## 📌 1. SVM with Different Kernels (Breast Cancer Dataset)

Dataset: Breast Cancer Dataset  
Algorithm: Support Vector Machine (SVM)  
Kernels Used:
- Linear
- RBF
- Polynomial
- Sigmoid

Evaluation Method:
- 5-Fold Cross Validation
- 10-Fold Cross Validation

Conclusion:
The Linear kernel achieved the highest accuracy (~95%), indicating that the dataset is nearly linearly separable.

---

## 📌 2. Feature Selection (Auto MPG Dataset)

Dataset: Auto MPG Dataset  
Target Variable: mpg (Regression Problem)

### a) Forward Selection
- Wrapper method
- Uses Linear Regression
- Selects features that improve R² score

### b) Backward Elimination
- Starts with all features
- Removes features that do not improve R² score

---

## 📌 3. Feature Importance

### a) Decision Tree Regressor
- Extracted feature importance using feature_importances_

### b) Random Forest Regressor
- Ensemble method
- Provides more stable feature importance
- Importance values converted into percentage

---

## 🛠 Technologies Used
- Python
- Scikit-learn
- Pandas
- NumPy

---

## 🎯 Overall Conclusion

- Linear SVM performed best for classification.
- Displacement was the most important feature for predicting MPG.
- Random Forest provided more stable feature importance compared to Decision Tree.

