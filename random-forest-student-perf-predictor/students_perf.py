#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 20:17:40 2019

@author: macbook
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("StudentsPerformance.csv")
X = data.iloc[:, [1,2,4,5]].values
y = data.iloc[:, 0].values
labelencoder = LabelEncoder()
X[:, 0] = labelencoder.fit_transform(X[:, 0])
X[:, 1] = labelencoder.fit_transform(X[:, 1])
X[:, 3] = labelencoder.fit_transform(X[:, 3])
y = labelencoder.fit_transform(y)

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=1000, criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


