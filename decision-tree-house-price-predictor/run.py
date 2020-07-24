import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

def split_price(row):
    prices = row['Price'].split(" ")
    price = prices[1].replace(",", "")
    return price

#DATA PREPROCESSING
data = pd.read_csv("data_kaggle.csv")

data = data.dropna()
data = data.reset_index(drop=True)

data['Price_New'] = data.apply(split_price, axis=1)

one_hot_loc = pd.get_dummies(data['Location'])
data = data.join(one_hot_loc)

one_hot_rooms = pd.get_dummies(data['Rooms'])
data = data.join(one_hot_rooms)

one_hot_type = pd.get_dummies(data['Property Type'])
data = data.join(one_hot_type)

one_hot_size = pd.get_dummies(data['Size'])
data = data.join(one_hot_size)

le = LabelEncoder()
data['Furnishing_enc'] = le.fit_transform(data['Furnishing'])

X = data.iloc[:, 8:].values
print(len(X))
data['Price'] = data['Price_New'].astype(int)
y = data['Price'].values
print(len(y))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

reg = RandomForestRegressor(n_estimators=100, random_state=0)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

"""Mean Absolute Error: 3579.234772204806
Mean Squared Error: 29228219358.370502
Root Mean Squared Error: 170962.6256184974"""



