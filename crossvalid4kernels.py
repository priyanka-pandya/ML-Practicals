#CROSS VALIDATION WITH FOUR KERNELS USES SVM,CV=5,CV=10 WITH BREAST CANCER DATASET
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

# Load Breast Cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target

# List of kernels
kernels = ['linear', 'rbf', 'poly', 'sigmoid']

# Loop through each kernel
for kernel in kernels:
    print("Kernel:", kernel)
    
    model = SVC(kernel=kernel)
    
    # 5-Fold Cross Validation
    scores_5 = cross_val_score(model, X, y, cv=5)
    print("5-Fold CV Accuracy:", scores_5.mean())
    
    # 10-Fold Cross Validation
    scores_10 = cross_val_score(model, X, y, cv=10)
    print("10-Fold CV Accuracy:", scores_10.mean())
