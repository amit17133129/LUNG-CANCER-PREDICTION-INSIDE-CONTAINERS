

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
  <img width="900" height="150" src="https://miro.medium.com/max/1094/1*QLGlwYe2p-5mDo96_RucyA.jpeg">
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
  <img width="900" height="200" src="https://miro.medium.com/max/1094/1*Xp5PjuNdUTEIFOfZqGm8mQ.jpeg">
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

```
