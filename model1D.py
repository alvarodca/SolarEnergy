import pandas as pd
import numpy as np
from numpy import mean, std
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

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

# Some useful information for our Conv1D model

# The feature maps are the number of times the input is processed or interpreted, 
# whereas the kernel size is the number of input time steps considered as the input sequence is read or processed onto the feature maps.

# Layers
# Conv1D <- Learns local patterns along the sequences, parameters: 
# filters (feature maps)<- how many patters it learns, more filters more patterns but more cost
# kernel_size <- window length (how much points of delta n are processed)

# Pooling <- reduces the size
# MaxPooling1D <- Takes maximum values at each conv, params: pool_size <- window length

# BatchNormalization <-  Normalizes Activations

# Flatten <- Converts feature maps to 1D vector, connects Conv Layers to Dense ones

# Dense <- Learns combination of features extracted by the CNNs, units <- number of neurons, activation <- activation function

# Dropout <- percentage of dropped neurons to avoid overfitting

# Output Layer <- maps features to output units
# For binary use units=1 and activation = sigmoid
# For multi-class use units = number of classes and activation = softmax

n_timesteps, n_features, n_outputs = X_train.shape[1],1,3

# Initializing our model
model = Sequential()
model.add(Conv1D(32,kernel_size = 5, activation = "relu", input_shape = (n_timesteps, n_features)))
model.add(BatchNormalization())
model.add(MaxPooling1D(2))
model.add(Conv1D(64, kernel_size = 5, activation = "relu"))
model.add(BatchNormalization())
model.add(MaxPooling1D(2))
model.add(Conv1D(128, kernel_size = 5, activation = "relu"))
model.add(BatchNormalization())
model.add(MaxPooling1D(2))
model.add(Flatten())
model.add(Dense(128,activation = "relu"))
model.add(Dropout(0.4))
model.add(Dense(64, activation = "relu"))
model.add(Dropout(0.3))
model.add(Dense(n_outputs,activation = "softmax"))

model.compile(loss = "sparse_categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"]) 
# Sparse categorical cross entropy as we have integer labels, if not we have to one hot encode our labels 

batch_size, epochs = 32, 10 # Bath size is the number of processed curves

# Training
model.fit(X_train, y_train, epochs = 10, batch_size = 32, verbose = 1)
# Verbose its for printing output throught the process, might slow down the process but useful, if set to 0, does not print

# Evaluation
loss ,accuracy = model.evaluate(X_test, y_test, batch_size = batch_size, verbose = 1)
print("Loss:", loss,"Accuracy:", accuracy)
# Saving the model
model.save("conv1d.keras")

# Loading the model
model = load_model("conv1d.keras")




# individual curves prediction
# y_pred = model.predict(X_test)
# y_pred_classes = np.argmax(y_pred, axis=1)  # conv<ert softmax output to class index

# from sklearn.metrics import confusion_matrix, classification_report

# print("Confusion Matrix: \n",confusion_matrix(y_test, y_pred_classes))
# print("Classification Report: \n",classification_report(y_test, y_pred_classes))
