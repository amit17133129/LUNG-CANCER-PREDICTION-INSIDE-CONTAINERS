

<p align="center">
             <center> <h1> LUNG CANCER PREDICTION INSIDE CONTAINERS 🐋  </h1> </center>
</p>

<p align="center">
  <img width="900" height="400" src="https://miro.medium.com/max/3840/1*6ORJX1A5NYom1ClGa7xwjQ.jpeg">
</p>

Hello guys! Back with another article. In this article i will go through how we can train machine learning models inside containers. For containers technology i will be using docker.

# Why Docker 🐋🐋 ??

Because Docker containers encapsulate everything an application needs to run (and only those things), they allow applications to be shuttled easily between environments. Any host with the Docker runtime installed — be it a developer’s laptop or a public cloud instance — can run a Docker container.
In this article, i will be using lung cancer dataset. This would be `classification problem`. On the given inputs the prediction will be a particular guys whether he is affected with lung cancer or not.


In this article, i will be using lung cancer dataset. This would be classification problem. On the given inputs the prediction will be a particular guys whether he is affected with lung cancer or not.

<p align="center">
  <img width="900" height="150" src="https://miro.medium.com/max/1094/1*ig1fQCpMMyKqA2-1prrKFw.jpeg">
</p>

```
docker    -it    --name   lung_cancer_os   centos:7
```

So, i have launched docker on AWS inside ec2 instance. You have to install python3 inside docker. Here os name is `lung_cancer_os`.

<p align="center">
  <img width="900" height="100" src="https://miro.medium.com/max/1094/1*QBZIQWmDSvV3XSA4Kbtcew.jpeg">
</p>

```
yum install python3 -y
```

Now you have to create a `requirement.txt` file same as blow. In this file you have to mention how many libraries you wanted to install. This process is less tike taking and can easily install the libraries.
<p align="center">
  <img width="900" height="150" src="https://miro.medium.com/max/1094/1*hg17URIYWEU6KSxJY90L3Q.jpeg">
</p>

```
pandas
numpy
scikit-learn
joblib
```

Now you have to install all these libraries.
<p align="center">
  <img width="900" height="150" src="https://miro.medium.com/max/1094/1*dDUyvb_tIIx4XYWmdnCtTQ.jpeg">
</p>

After installing you you can check out the list of all the libraries installed.

```
pip3 list
```

<p align="center">
  <img width="900" height="250" src="https://miro.medium.com/max/1094/1*QLGlwYe2p-5mDo96_RucyA.jpeg">
</p>

Now we have to read lung_cancer data and predict on different classification algorithms.
# Importing Required Libraries:

```
import numpy as np   # importing numpy for numerical calculations
import pandas as pd # importing  pandas for creating data frames
from sklearn.model_selection import train_test_split  # train_tets_split for spliting the data into training an testing
from sklearn.linear_model import LogisticRegression  # for logistic resgression
from sklearn.ensemble import RandomForestClassifier # for random 
forest classifierfrom sklearn.ensemble import GradientBoostingClassifier     # For gradienboosting classifier
from sklearn.metrics import accuracy_score # importing metrics for measuring accuracy
from sklearn.metrics import mean_squared_error  # for calculating mean squre errors
```
# Reading Lung Cancer Data:
Now we will be reading lung cancer data using `pandas.read_csv()` function

```
df = pd.read_csv("lung_cancer_examples.csv")  # reading csv data
print(df.sample)
```

<p align="center">
  <img width="900" height="150" src="https://miro.medium.com/max/1094/1*Xp5PjuNdUTEIFOfZqGm8mQ.jpeg">
</p>

In the above output the column names are not in the sequence as see that, this is because i have used print(df) inside docker.
Now we have to select the features and divide then into training and testing data.

```
X = df.drop[['Name', 'Surname', 'Result']]
y = df.iloc[:, -1]
```
As there were less feature and anyone can understand that `names` and `surnames` are not much important to predict `lung cancer`, also `Results` is not independent feature therefore, i removed them all and rest will be kept as a features inside `X` variable. The `y` (dependent variable) will contain only the Results feature.
Now we have to divide the data into training and testing part. so here we will use `train_test_split` function from sklearn library. here i am taking `training data as 80%` and `testing data as 20%`.

```
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=10)
```

<p align="center">
  <img width="900" height="200" src="https://miro.medium.com/max/1094/1*A-9Ui59K21OR_9fqy73Qvg.gif">
</p>

Now we are ready to train the different classification models. Here we are taking first as *logistic regression* then *gradient boosting classifer* followed by *random forest classifier*.

# Logistic Regression:
LogisticRegression is meant for classification problem. As we also having a classification dataset so we can implement classification easily. LogisticRegression uses a sigmoid function which helps it to train the model on large dataset. To use LogisticRegression() funtion we have to import linear_model module from sklearn library.

```
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
```

Inside logistic_regression class I have created three functions i.e., logistic(), mean_absolute_error() and variance_bias().
logistic_regression() function will train the model using LogisticRegession() function and it will return the model accuracy. This function accepts parameters X_train, y_train (training data). Inside this function model_lr = LogisticRegession() will create a model and save it into model_lr variable and then model_lr.fit(X_train, y_train) will train the model and after training we have to use testing data i.e., X_test for prediction for checking the accuracy of the model using accuracy_score() function.
mean_absolute_error() function will return the error generated while doing the prediction. This function will use np.square(y_test — y_pred_lr).mean()) formula. First it will square the differences between y_test(actual value) and y_pred_lr(predicted value) and it will take a mean of the value obtained by taking the square. Here error = y_test — y_pred_lr.
variance_bias() function will helps to obtain the bias and variance of the model. Variance is obtained using var() function from numpy and it will obtain variance on predicted values i.e., y_pred_lr. Now to obtain the bias we have to take help from variance value. First we have to calculate square of error using SSE = np.mean((np.mean(y_pred_lr) — y_test)** 2) formula. First this formula will take mean of (y_pred_lr) and then calculate the error mean(y_pred_lr) — y_test) and then it will square the values. Now variance’s value will be subtracted from SSE and it will give bias value. The less the value of the variance will have less variety in data. The goal is to balance the bias and variance, so the model does not overfit or underfit. If the variance and bias go high then it will affect the model accuracy. Every machine learning algorithm will give difference values of bias and variance.

# GradientBoostingClassifer:

GB builds an additive model in a forward stage-wise fashion; it allows for the optimization of arbitrary differentiable loss functions. In each stage n_classes_ regression trees are fit on the negative gradient of the binomial or multinomial deviance loss function. Binary classification is a special case where only a single regression tree is induced.
```
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
```


Inside gradient_boosting class I have created three functions i.e., gb(), mean_absolute_error() and variance_bias().

gb() function will train the model using GradientBoostingClassifier() function and it will return the model accuracy. This function accepts parameters X_train, y_train (training data). Inside this function model_gbc = GradientBoostingClassifier() will create a model and save it into model_gbc variable and then model_gbc.fit(X_train, y_train) will train the model and after training we have to use testing data i.e., X_test for prediction for checking the accuracy of the model using accuracy_score() function.

mean_absolute_error() function will return the error generated while doing the prediction. This function will use np.square(y_test — y_pred_gbc).mean()) formula. First it will square the differences between y_test(actual value) and y_pred_rf(predicted value) and it will take a mean of the value obtained by taking the square. Here error = y_test — y_pred_gbc.

variance_bias() function will helps to obtain the bias and variance of the model. Variance is obtained using var() function from numpy and it will obtain variance on predicted values i.e., y_pred_gbc. Now to obtain the bias we have to take help from variance value. First we have to calculate square of error using SSE = np.mean((np.mean(y_pred_gbc) — y_test)** 2) formula. First this formula will take mean of (y_pred_gbc) and then calculate the error mean(y_pred_gbc) — y_test) and then it will square the values. Now variance’s value will be subtracted from SSE and it will give bias value. The less the value of the variance will have less variety in data. The goal is to balance the bias and variance, so the model does not overfit or underfit. If the variance and bias go high then it will affect the model accuracy.

# RandomForestClassifier:

RandomForestClassifier is meant for classification problem. As we also having a classification dataset so we can implement classification easily. RandomForestClassifier uses Decision tree classifier for create trees and group of trees will create forests and it will help us to train then model and give accuracy results. Trees are helpful in taking decision. If the depth of trees increases then the decision(accuracy) will be more accurate. To use RandomForestClassifier() funtion we have to import ensemble module from sklearn library. Ensemble means the algorithms are assembled into a group and that module name is ensemble. We have more ensemble algorithms e.g., AdaBoostClassifier, CatBoostClassifer, GradientBoostingClassifier.

```
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
```

Inside random_forest_classifier class I have created three functions i.e., random_forest(), mean_absolute_error() and variance_bias().

random_forest() function will train the model using RandomForestClassifier() function and it will return the model accuracy. This function accepts parameters X_train, y_train (training data). Inside this function model_rf = RandomForestClassifier() will create a model and save it into model_rf variable and then model_rf.fit(X_train, y_train) will train the model and after training we have to use testing data i.e., X_test for prediction for checking the accuracy of the model using accuracy_score() function.

mean_absolute_error() function will return the error generated while doing the prediction. This function will use np.square(y_test — y_pred_rf).mean()) formula. First it will square the differences between y_test(actual value) and y_pred_rf(predicted value) and it will take a mean of the value obtained by taking the square. Here error = y_test — y_pred_rf.

variance_bias() function will helps to obtain the bias and variance of the model. Variance is obtained using var() function from numpy and it will obtain variance on predicted values i.e., y_pred_rf. Now to obtain the bias we have to take help from variance value. First we have to calculate square of error using SSE = np.mean((np.mean(y_pred_rf) — y_test)** 2) formula. First this formula will take mean of (y_pred_rf) and then calculate the error mean(y_pred_rf) — y_test) and then it will square the values. Now variance’s value will be subtracted from SSE and it will give bias value. The less the value of the variance will have less variety in data. The goal is to balance the bias and variance, so the model does not overfit or underfit. If the variance and bias go high then it will affect the model accuracy.

# Training and Testing Models:

Now we have to train and test the logistic regression, random forest classifier and gradient boosting model.

```
print("-------LUNG CANCER PREDICTION USING LOGISTIC REGRESSION--------")
# calling the class logistic_regression and creating object.
logistic = logistic_regression()
# calling logistic function that accepts two parameters i.e X_train, y_train
print(logistic.logistic(X_train, y_train))
# getting accuracy of logistic regression model
print(logistic.mean_absolute_error())        # getting mean absolute error
print(logistic.variance_bias())              # getting variance and bias
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
~
```
<p align="center">
  <img width="900" height="400" src="https://miro.medium.com/max/1094/1*1n_ec0E96HwHPKkfw9wuzQ.jpeg">
</p>

From above results we can conclude that logistic regression is behaving excellent followed by random forest classifier and gradient boosting classifier. This is due to less amount of data therefore, its giving 100 results. hope you like this article and comment will be appriciated. 😊😊




