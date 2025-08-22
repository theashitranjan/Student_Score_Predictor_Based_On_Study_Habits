# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 10:05:38 2025

@author: ashit
"""

# importing modules and libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error

# loading dataset

df = pd.read_csv('student_scores.csv')

# defining features
X = df[['Hours_Studied','Attendance']]
y = df['Final_Score']

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Model training

model = LinearRegression()
model.fit(X_train, y_train)

print('----------------Model trained successfullly---------------')

# Interactive application

while True:
    print("type 'exit' to close the program")
    hours_input = input('Enter the study hours: ')
    if hours_input.lower() == 'exit':
        break
    attendance_input = input('Enter attendance percentage: ')
    if attendance_input.lower() == 'exit':
        break
    
    # datatype changing
    new_hrs = float(hours_input)
    new_attendance = float(attendance_input)
    
    # Prediction 
    predicted = model.predict([[new_hrs, new_attendance]])
    
    print('Predicted score is: ',predicted)
    

