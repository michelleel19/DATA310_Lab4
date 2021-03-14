## DATA 310 Lab 4

### Question 1
Regularization is defined as:

Answer: The minimization of the sum of squared residuals subject to a constraint on the weights (aka coefficients).

### Question 2
The regularization with the square of an L2 distance may improve the results compared to OLS when the number of features is higher than the number of observations.

Answer: True, because regularization penalizes large weights and biases.

### Question 3
The L1 norm always yields shorter distances compared to the Euclidean norm.

Answer: False (L2 is the hypotenuse of the triangle, so L1 is the other sides of the triangle)

### Question 4
Typically, the regularization is achieved by

Answer: minimizing the average of the squared residuals plus a penalty function whose input is the vector of coefficients.

### Question 5
A regularization method that facilitates variable selection (estimating some coefficients as zero) is 

Answer: Lasso (Ridge can't zero out coefficients)

### Question 6
Write your own Python code to import the Boston housing data set (from the sklearn library) and scale the data (not the target) by z-scores. If we use all the features with the Linear Regression to predict the target variable then the root mean squared error (RMSE) is:
```markdown
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as MSE
from sklearn.preprocessing import StandardScaler

data = load_boston()
X = pd.DataFrame(data=data.data, columns=data.feature_names)
y = data.target

scale = StandardScaler() 
model = LinearRegression()
model.fit(Xs, y) 
y_pred = model.predict(Xs) 

rmse = np.sqrt(MSE(y,y_pred))
print(rmse)
```
Answer: 4.6791
### Question 7
On the Boston housing data set if we consider the Lasso model with 'alpha=0.03' then the 10-fold cross-validated prediction error is:
(for the 10-fold cross-validation shuffle you should use random_state=1234, your final answer should include only the first 4 decimals that you get from the code)
```markdown
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from yellowbrick.regressor import ResidualsPlot
from sklearn.pipeline import Pipeline

from sklearn.model_selection import KFold # import KFold
kf = KFold(n_splits=10, random_state=1234,shuffle=True)

data = load_boston()
df = pd.DataFrame(data=data.data, columns=data.feature_names)
y = data.target

i = 0
PE = []
model = Lasso(alpha=0.03)
for train_index, test_index in kf.split(df):
    X_train = df.values[train_index]
    y_train = y[train_index]
    X_test = df.values[test_index]
    y_test = y[test_index]
    model = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    PE.append(np.sqrt(MSE(y_test, y_pred)))
    # print('RMSE from each fold:',np.sqrt(MSE(y_test, y_pred)))

print('The k-fold crossvalidated prediction error is: ' + str(np.mean(PE)))
```
Answer: 4.8370

### Question 8
On the Boston housing data set if we consider the Elastic Net model with 'alpha=0.05' and 'l1_ratio=0.9' then the 10-fold cross-validated prediction error is:
(for the 10-fold cross-validation shuffle you should use random_state=1234, your final answer should include only the first 4 decimals that you get from the code)
```markdown
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from yellowbrick.regressor import ResidualsPlot
from sklearn.pipeline import Pipeline

from sklearn.model_selection import KFold # import KFold
kf = KFold(n_splits=10, random_state=1234,shuffle=True)

data = load_boston()
df = pd.DataFrame(data=data.data, columns=data.feature_names)
y = data.target

i = 0
PE = []
model = ElasticNet(alpha=0.05,l1_ratio = 0.9)
for train_index, test_index in kf.split(df):
    X_train = df.values[train_index]
    y_train = y[train_index]
    X_test = df.values[test_index]
    y_test = y[test_index]
    model = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    PE.append(np.sqrt(MSE(y_test, y_pred)))
    # print('RMSE from each fold:',np.sqrt(MSE(y_test, y_pred)))

print('The k-fold crossvalidated prediction error is: ' + str(np.mean(PE)))
```
Answer: 4.8965

### Question 9
If we create all quadratic polynomial (degree=2) features based on the z-scores of the original features and then apply OLS, the root mean squared error is:
```markdown
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

data = load_boston()
df = pd.DataFrame(data=data.data, columns=data.feature_names)
y = data.target

scale = StandardScaler()
Xs = scale.fit_transform(df)

polynomial_features= PolynomialFeatures(degree=2)
x_poly = polynomial_features.fit_transform(Xs)

model = LinearRegression()
model.fit(x_poly, y)
y_poly_pred = model.predict(x_poly)

rmse = np.sqrt(mean_squared_error(y,y_poly_pred))
print(rmse)
```
Answer: 2.4483

### Question 10
If we create all quadratic polynomial (degree=2) features based on the z-scores of the original features and then apply the Ridge regression with alpha=0.1 and we create a Quantile-Quantile plot for the residuals then the result shows that  the obtained residuals pretty much follow a normal distribution.

Answer: True
