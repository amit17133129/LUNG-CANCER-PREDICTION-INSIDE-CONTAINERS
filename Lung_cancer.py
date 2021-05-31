import numpy as np   # importing numpy for numerical calculations

import pandas as pd # importing  pandas for creating data frames
from sklearn.model_selection import train_test_split  # train_tets_split for spliting the data into training an testing
 from sklearn.linear_model import LogisticRegression  # for logistic resgression
from sklearn.ensemble import RandomForestClassifier # for random forest classifierfrom sklearn.ensemble import GradientBoostingClassifier     # For gradienboosting classifier
from sklearn.metrics import accuracy_score # importing metrics for measuring accuracy
from sklearn.metrics import mean_squared_error  # for calculating mean squre errors

df = pd.read_csv("lung_cancer_examples.csv")  # reading csv data
print(df.sample)

X = df.drop[['Name', 'Surname', 'Result']]
y = df.iloc[:, -1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=10)

class logistic_regression:    # creating a logistic regression class
    def logistic(self, X_train, y_train):    # creating a function that will create a model and after training it will give accuracy
         # Here we are creating a decision tree model using LogisticRegression. to do that we have to fit the training data (X_train, y_rain) into model_lr object
         from sklearn.linear_model import LogisticRegression
         model_lr = LogisticRegression()    # creating a logistic model
         model_lr.fit(X_train, y_train)
         # here we are predicting the Logistic Regression model on X_test [testing data]
         self.y_pred_lr = model_lr.predict(X_test)
         #print("Mean square error for logistic regression model: ", mean_squared_error(y_test, y_pred_lr)) # will give mean square error of the model
         # accuracy_score will take y_test(actual value) and y_pred_lr(predicted value) and it will give the accuracy of the model
         print("Logistic Regression model Accuracy               :", accuracy_score(y_test, self.y_pred_lr)*100, "%")
    def mean_absolute_error(self):
         print("Mean Absoluter error of logistic Regression     :", np.square(y_test - self.y_pred_lr).mean()) # calculating mean absolute error of LogisticRegression model

    def variance_bias(self):
        Variance = np.var(self.y_pred_lr)     # calculating variance in the predicted output
        print("Variance of LogisticRegression model is         :", Variance)
        SSE = np.mean((np.mean(self.y_pred_lr) - y_test)** 2)  # calculating s=sum of square error
        Bias = SSE - Variance                         # calculating Bias taking a difference between SSE and Variance
        print("Bias of LogisticRegression model is             :", Bias)

class gradient_boosting:    # creating a logistic regression class
    def gb(self, X_train, y_train):    # creating a function that will create a model and after training it will give accuracy
         # Here we are creating a decision tree model using LogisticRegression. to do that we have to fit the training data (X_train, y_rain) into model_lr object
         from sklearn.ensemble import GradientBoostingClassifier
         model_gbc = GradientBoostingClassifier()    # creating a logistic model
         model_gbc.fit(X_train, y_train)
         # here we are predicting the Logistic Regression model on X_test [testing data]
         self.y_pred_gbc = model_gbc.predict(X_test)
         #print("Mean square error for logistic regression model: ", mean_squared_error(y_test, y_pred_lr)) # will give mean square error of the model
         # accuracy_score will take y_test(actual value) and y_pred_lr(predicted value) and it will give the accuracy of the model
         print("Logistic Regression model Accuracy               :", accuracy_score(y_test, self.y_pred_gbc)*100, "%")
    def mean_absolute_error(self):
         print("Mean Absoluter error of logistic Regression     :", np.square(y_test - self.y_pred_gbc).mean()) # calculating mean absolute error of LogisticRegression model

    def variance_bias(self):
        Variance = np.var(self.y_pred_gbc)     # calculating variance in the predicted output
        print("Variance of LogisticRegression model is         :", Variance)
        SSE = np.mean((np.mean(self.y_pred_gbc) - y_test)** 2)  # calculating s=sum of square error
        Bias = SSE - Variance                         # calculating Bias taking a difference between SSE and Variance
        print("Bias of LogisticRegression model is             :", Bias)

class random_forest_classifier:
    def random_forest(self, X_train, y_train):
               # Here we are creating a decision tree model using RandomForestClassifier. to do that we have to fit the training data (X_train, y_rain) into model_rc object
               from sklearn.ensemble import RandomForestClassifier
               self.model_rf = RandomForestClassifier()
               self.model_rf.fit(X_train, y_train)
               # here we are predicting the  Random Forest Classifier model on X_test [testing data]
               self.y_pred_rf = self.model_rf.predict(X_test)
               print("Mean square error for random forest model: ", mean_squared_error(y_test, self.y_pred_rf)) # will give mean square error of the model
               # accuracy_score will take y_test(actual value) and y_pred_rc(predicted value) and it will give the accuracy of the model
               print("Random Forest model accuracy              :",  accuracy_score(y_test, self.y_pred_rf)*100, "%")
   
    def mean_absolute_error(self):
               print("Mean Absoluter error of Random Forest     :", np.square(y_test - self.y_pred_rf).mean()) # calculating mean absolute error of RandomForest model
    def variance_bias(self):
               Variance = np.var(self.y_pred_rf)     # calculating variance in the predicted output
               print("Variance of RandomForest model is         :", Variance)
               SSE = np.mean((np.mean(self.y_pred_rf) - y_test)** 2)  # calculating s=sum of square error
               Bias = SSE - Variance                         # calculating Bias taking a difference between SSE and Variance
               print("Bias of RandomForest model is             :", Bias)

print("-------LUNG CANCER PREDICTION USING LOGISTIC REGRESSION--------")
# calling the class logistic_regression and creating object.
logistic = logistic_regression()
# calling logistic function that accepts two parameters i.e X_train, y_train
print(logistic.logistic(X_train, y_train))
# getting accuracy of logistic regression model
print(logistic.mean_absolute_error())        # getting mean absolute error
print(logistic.variance_bias())           # getting variance and bias
print("-------LUNG CANCER PREDICTION USING GRADIENT BOOSTING CLASSIFIER--------")
# calling the class gradient_boosting and creating object.
gbc = gradient_boosting()
# calling gb function that accepts two parameters i.e X_train, y_train
print(gbc.gb(X_train, y_train))
# getting accuracy of GradientBoostingClassifier model
print(gbc.mean_absolute_error())        # getting mean absolute error
print(gbc.variance_bias())              # getting variance and bias
print("-------LUNG CANCER PREDICTION USING RANDOM FOREST CLASSIFIER--------")
# calling the class random_forest_classifier and creating object.
rf_classifier = random_forest_classifier()
# calling random_forest function that accepts two parameters i.e X_train, y_train
print(rf_classifier.random_forest(X_train, y_train)) # getting accuracy of of random forest model
print(rf_classifier.mean_absolute_error())     # getting mean absolute error
print(rf_classifier.variance_bias())           # getting variance and bias
