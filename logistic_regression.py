

## Advertisement Sale Prediction from Existing customer using **Logistic Regression**
"""
**Importing libraries**
"""

import pandas as pd
import numpy as np

"""**Choose dataset from files**"""

from google.colab import files
uploaded = files.upload()

"""**loading dataset**"""

dataset = pd.read_csv("DigitalAd_dataset.csv")

"""**Summarize Dataset**"""

print(dataset.shape)
print(dataset.head(3))

"""**Segregrating Dataset into Independent varaiable(X) and dependent Varaiable(Y)**"""

X = dataset.iloc[:,:-1]
print(X)

Y = dataset.iloc[:,-1]
print(Y)

"""**Splitting Dataset into Train and Test**"""

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.25,random_state=0)

"""**Transforming Data**"""

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train=sc.fit_transform(x_train)
X_test = sc.transform(x_test)

"""**Training**"""

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state=0)
model.fit(X_train,y_train)

"""**Predicting whether new customer with age and salary will buy or not**"""

age = int(input("Enter your age"))
salary = int(input("Enter your salary"))
newCust = [[age,salary]]
result = model.predict(sc.transform(newCust))
print(result)
if result == 1:
  print("Customer will buy")

"""**Prediction for test data**"""

y_pred =model.predict(x_test)
print(y_pred)

"""**Evaluating model performance**"""

from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test,y_pred)
print("Confusion Matrix")
print(cm)
accuracy = accuracy_score(y_test,y_pred)*100
print("Accuracy of model {accuracy}")