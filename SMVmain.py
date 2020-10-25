import sklearn
from sklearn import datasets
from sklearn import svm

cancer_data = datasets.load_breast_cancer()

print(cancer_data.feature_names)
print(cancer_data.target_names)

x = cancer_data.data
y = cancer_data.target

#training the algorithm to handle the data
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

#Results that will be returned (y value)
classes = ['malignant' 'benign']

#implementing our classifier
classifier = svm.SVC()