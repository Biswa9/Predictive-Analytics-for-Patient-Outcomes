#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install tensorflow


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib

# Load and prepare data
data = pd.read_csv('patient_data.csv')
data.fillna(data.mean(), inplace=True)

# Encode categorical variables
label_encoders = {}
for column in ['Gender', 'Diagnosis']:  # Replace with actual categorical columns
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Split the dataset into features and target
X = data.drop('ReadmissionRisk', axis=1)  # Replace 'ReadmissionRisk' with your target variable
y = data['ReadmissionRisk']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Logistic Regression model
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# Evaluate Logistic Regression model
print("Logistic Regression Model Evaluation:")
print(confusion_matrix(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

# Train Neural Network model
nn_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = nn_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# Evaluate Neural Network model
loss, accuracy = nn_model.evaluate(X_test, y_test)
print(f"\nNeural Network Model Evaluation:\nLoss: {loss}, Accuracy: {accuracy}")

# Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

# Save models and scaler
joblib.dump(lr_model, 'readmission_risk_lr_model.pkl')
joblib.dump(nn_model, 'readmission_risk_nn_model.h5')
joblib.dump(scaler, 'scaler.pkl')

print("Models and scaler have been saved successfully.")


# In[ ]:




