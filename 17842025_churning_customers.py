# -*- coding: utf-8 -*-
"""17842025_churning_customers.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Tvfj6gOv3Oyzs1wO7q71FDSO_V_hDxIy

Importation of Libraries

PREPROCESSING
"""

from google.colab import drive
drive.mount('/content/drive')

#Customer churn refers to the loss of customers or subscribers for any reason at all

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

"""Reading the csv file"""

churn_file=pd.read_csv('/content/drive/MyDrive/Colab Notebooks/CustomerChurn_dataset.csv')
churn_file

"""

Exploratory Data Analysis"""

#Checking if there is any null values
churn_file.isnull().sum()

churn_file.describe()

# Identify columns that can be converted to float

df = pd.DataFrame(churn_file)
columns_to_convert = []

for col in df.columns:
    try:
        df[col].astype(float)
        columns_to_convert.append(col)
    except ValueError:
        pass
df.isnull().sum()

#Dropping useless columns; customerID in this case
df = df.drop('customerID', axis=1)
df

#Totalcharge is object instead of float.
#Converting non-numeric values to numeric values.
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
# Fill NaN values with the mean

mean_total_charges = df['TotalCharges'].mean()
df['TotalCharges'] = df['TotalCharges'].fillna(mean_total_charges)
df

"""converts categorical variables into a numerical

Encoding
"""

# Get columns that are numeric
numeric_columns = ['MonthlyCharges', 'TotalCharges', 'SeniorCitizen', 'tenure']

# Assuming 'churn_file' is your DataFrame
columns_to_encode = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
                     'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                     'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'Churn']

# Encoding the non-numeric attributes
for column in columns_to_encode:
    df[column], _ = pd.factorize(df[column])

# Display the resulting DataFrame
df

"""Finding correlation"""

#Finidng correlation
correlation_matrix = df.corr()

# Generate a heatmap of the correlation matrix
plt.figure(figsize=(10, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix Heatmap')
plt.show()

# Specify the features
features = df.columns
features

"""Extracting a highly correlated columns"""

# Find highly correlated variables
highly_corr = set()
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.5 :
            colname = correlation_matrix.columns[i]
            highly_corr.add(colname)

# Remove one of the highly correlated variables
if len(highly_corr) > 0:
    variable_to_remove = list(highly_corr)[0]
    new_Data = df.drop(columns=[variable_to_remove])
    print(f"Removed variable: {variable_to_remove}")

new_Data

"""Relevant Features

Visualization with histogram
"""

#Visualization
#Histogram for the housing data
df.hist(bins=50, figsize=(10, 15))
plt.show()

df = df.dropna()

"""
Training an MLP using features from (1) with cross validation and GridSearchCV"""

# Separating the dependend and independent datasets

y_values = df['Churn']
x_values = df.drop('Churn', axis = 1)

columns_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges']

x_value = x_values.drop(columns=columns_to_scale)

X = df[columns_to_scale]

X.shape

from sklearn.preprocessing import StandardScaler

# Initialize the scaler
scaler = StandardScaler()

# Fit the scaler on the training data and transform both training and test data
'''columns_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges']

X_train1 = X_train.drop(columns=columns_to_scale)
X_test1 = X_test.drop(columns=columns_to_scale)'''

X_scaled= scaler.fit_transform(X)
Final_X= pd.concat([X, x_value], axis=1)

"""independent column values

dependent values
"""

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(Final_X, y_values, test_size=0.2, random_state=42, stratify=y_values)

"""Creating a model"""

input_layer = Input(shape=(X_train.shape[1],))

hidden_layer_1 = Dense(128, activation='relu')(input_layer)
hidden_layer_2 = Dense(64, activation='relu')(hidden_layer_1)
hidden_layer_3 = Dense(32, activation='relu')(hidden_layer_2)
hidden_layer_4 = Dense(8, activation='relu')(hidden_layer_3)

output_layer = Dense(1, activation='sigmoid')(hidden_layer_4)

# Creating the model
model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model using your training data
model.fit(X_train, y_train, epochs=10, batch_size=64)

# Evaluate the model on your test data
loss, accuracy = model.evaluate(X_test, y_test)

# Evaluate the model on your test data
loss, accuracy = model.evaluate(X_test, y_test)

# Evaluate the model on the test set
y_prediction = model.predict(X_test)
auc_score = roc_auc_score(y_test, y_prediction)

auc_score

# Displaying the model summary
model.summary()

#Using a functional API keras

# Create the Keras model
model = keras.Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model using your training data
model.fit(X_train, y_train, epochs=15, batch_size=32)


# Predict probabilities on the test
y_pred_prob = model.predict(X_test)

#AUC score
auc_score = roc_auc_score(y_test, y_pred_prob)
print(f"AUC Score: {auc_score:.4f}")

# Evaluate the model on your test data
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

"""Testing the model"""

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


# Define the MLP model as a function
def multi_layer_perceptron(input_size, hidden_sizes, output_size):
    model = MLPClassifier(
        hidden_layer_sizes=hidden_sizes,
        activation='relu',
        max_iter=500
    )
    return model


input_size = 19
hidden_layers = (128,), (64,), (32,), (16,), (8,), (4,), (2,), (64, 32), (32, 16)
output_size = 19

# Define the hyperparameter grid for the grid search
param_grid = {
    'hidden_layer_sizes': [(128,), (64,), (32,), (16,), (8,), (4,), (2,), (64, 32), (32, 16)],
}

# Create the MLP model
mlp_model = multi_layer_perceptron(input_size, hidden_layers, output_size)

# Create the GridSearchCV object with AUC score as the scoring metric
grid_search = GridSearchCV(mlp_model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)

# Perform the grid search on the training data
grid_search.fit(X_train, y_train)

# Get the best model from the grid search
best_mlp_model = grid_search.best_estimator_

# Make predictions on the test set using the best model
y_pred_proba = best_mlp_model.predict_proba(X_test)[:, 1]

# Evaluate the best model's AUC score on the test set
auc_score = roc_auc_score(y_test, y_pred_proba)

#final AUC score on the test set
print(auc_score)

"""Saving the model"""

import pickle

# Save the model
with open('Customer_churning.pkl', 'wb') as file_name:
    pickle.dump(best_mlp_model, file_name)