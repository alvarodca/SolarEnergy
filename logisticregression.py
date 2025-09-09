from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Obtaining data
df = pd.read_csv("metadata.csv")

# Grouping the points into curves
curves = df.groupby("curve_id")["SRH"].apply(list).to_list()

# Obtaining the labels
labels = df.groupby("curve_id")["Label"].first().to_numpy()

# Padding to an equal size
padded_curves = pad_sequences(curves, padding = "post", dtype = "float32") # Padding can be pre or post

# We have to add another dimension for features
X = np.expand_dims(padded_curves, -1)

y = labels

# Data Splitting
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1, random_state=42)

# Our logistic regression model expects just a 2D tensor (samples, feaures)
x_train_adjusted = X_train.reshape(X_train.shape[0], -1)
x_test_adjusted = X_test.reshape(X_test.shape[0],-1)

# Simple LR model
model = LogisticRegression(max_iter=1000)

# Training
model.fit(x_train_adjusted,y_train)

# Prediction
y_pred = model.predict(x_test_adjusted)

# Accuracy
accuracy = accuracy_score(y_test,y_pred)    
print("Accuracy: ", accuracy)

