# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 18:25:00 2025

@author: ashit
Description: An improved, error-resistant script to predict student scores.
"""

# importing modules and libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import sys # Used to exit the script if critical errors occur

# --- 1. Data Loading ---
# Use a try-except block to handle potential errors during file loading.
try:
    # loading dataset
    df = pd.read_csv('student_scores.csv')

    # defining features and target variable
    # This can raise a KeyError if the column names are wrong in the CSV.
    X = df[['Hours_Studied', 'Attendance']]
    y = df['Final_Score']

except FileNotFoundError:
    # This block runs if 'student_scores.csv' is not in the same folder.
    print("Error: 'student_scores.csv' not found. Please make sure the file is in the correct directory.")
    sys.exit() # Exit the program cleanly.

except KeyError as e:
    # This block runs if the required columns are missing from the CSV file.
    print(f"Error: A required column is missing from the CSV file: {e}")
    print("Please ensure the CSV contains 'Hours_Studied', 'Attendance', and 'Final_Score' columns.")
    sys.exit() # Exit the program cleanly.


# --- 2. Model Training and Evaluation ---

# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

print('---------------- Model trained successfully ---------------')

# Evaluate the model on the test set to understand its performance
# This gives an idea of how accurate the predictions are.
test_predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, test_predictions)
print(f"Model Performance (Mean Absolute Error on test data): {mae:.2f}")
print(f"This means the model's predictions are, on average, off by about {mae:.2f} points.")
print('-----------------------------------------------------------')


# --- 3. Interactive Application with Input Validation ---

while True:
    print("\nType 'exit' to close the program.")
    
    # Get user input for study hours
    hours_input = input('Enter the study hours: ')
    if hours_input.lower() == 'exit':
        break
    
    # Get user input for attendance
    attendance_input = input('Enter attendance percentage (e.g., 95): ')
    if attendance_input.lower() == 'exit':
        break
    
    try:
        # Convert inputs to floating-point numbers.
        # This will trigger the 'except' block if the user enters non-numeric text.
        new_hrs = float(hours_input)
        new_attendance = float(attendance_input)
        
        # Add logical validation for the inputs
        if new_hrs < 0:
            print("Warning: Study hours cannot be negative. Please try again.")
            continue # Skip the rest of this loop and ask for input again.
            
        if not (0 <= new_attendance <= 100):
            print("Warning: Attendance percentage must be between 0 and 100. Please try again.")
            continue # Skip the rest of this loop and ask for input again.

        # Prediction 
        # The model expects a 2D array, so we wrap our inputs in double brackets [[...]]
        predicted_score = model.predict([[new_hrs, new_attendance]])
        
        # Print the result, formatted to two decimal places for readability.
        print(f"Predicted score is: {predicted_score[0]:.2f}")

    except ValueError:
        # This block runs if float() fails because the input was not a valid number.
        print("Invalid input. Please enter only numeric values for hours and attendance.")

print("\nProgram closed. Goodbye!")
