import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import cross_val_score
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


from sklearn import metrics
import numpy as np

boston = fetch_openml(name = 'boston', version = 1, as_frame = True, parser = 'pandas')

df = boston.frame


df['price'] = boston.target
# Dividing the dataset into depdent and independent feature
x = df.iloc[:,: -1]
y = df.iloc[:,-1]



# Ensure all data in numeric (The fix)
X = x.astype(float)
Y = y.astype(float)

# Step 2 :- Initialize and run cross-validation
line_reg = LinearRegression()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state= 42)
line_reg.fit(X_train, Y_train)

prediction = line_reg.predict(X_test)



#1. R-Squared score ( Best possible score in 1.0)
r2 = metrics.r2_score(Y_test, prediction)

#2. Mean Absolute Error ( Average error in 'price' units)
mae = metrics.mean_absolute_error(Y_test, prediction)

#3. Root mean squared error ( Standard deviation of prediction errors)
rmse = np.sqrt(metrics.mean_squared_error(Y_test, prediction))

print(f"R-Squared Score: {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")