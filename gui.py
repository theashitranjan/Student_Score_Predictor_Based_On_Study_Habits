# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 12:30:00 2025

@author: ashit
"""

# importing modules and libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import customtkinter as ctk

# --- Machine Learning Model Setup ---
try:
    # loading dataset
    df = pd.read_csv('student_scores.csv')

    # defining features
    X = df[['Hours_Studied', 'Attendance']]
    y = df['Final_Score']

    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # Model training
    model = LinearRegression()
    model.fit(X_train, y_train)
    model_trained = True
    print('----------------Model trained successfully---------------')

except FileNotFoundError:
    model_trained = False
    print("Error: 'student_scores.csv' not found. Please make sure the file is in the correct directory.")
except Exception as e:
    model_trained = False
    print(f"An error occurred during model training: {e}")


# --- GUI Application ---

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        # --- Window Setup ---
        self.title("Student Score Predictor")
        self.geometry("450x350")
        self.grid_columnconfigure(0, weight=1) # Center content

        # --- Title ---
        self.title_label = ctk.CTkLabel(self, text="Predict Final Score", font=ctk.CTkFont(size=20, weight="bold"))
        self.title_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        # --- Input Frame ---
        self.input_frame = ctk.CTkFrame(self)
        self.input_frame.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        self.input_frame.grid_columnconfigure(1, weight=1)

        # Hours Studied Input
        self.hours_label = ctk.CTkLabel(self.input_frame, text="Hours Studied:")
        self.hours_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.hours_entry = ctk.CTkEntry(self.input_frame, placeholder_text="e.g., 8.5")
        self.hours_entry.grid(row=0, column=1, padx=10, pady=10, sticky="ew")

        # Attendance Input
        self.attendance_label = ctk.CTkLabel(self.input_frame, text="Attendance (%):")
        self.attendance_label.grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.attendance_entry = ctk.CTkEntry(self.input_frame, placeholder_text="e.g., 95")
        self.attendance_entry.grid(row=1, column=1, padx=10, pady=10, sticky="ew")

        # --- Predict Button ---
        self.predict_button = ctk.CTkButton(self, text="Predict Score", command=self.predict_score)
        self.predict_button.grid(row=2, column=0, padx=20, pady=10)

        # --- Result Display ---
        self.result_label = ctk.CTkLabel(self, text="", font=ctk.CTkFont(size=16))
        self.result_label.grid(row=3, column=0, padx=20, pady=20)

        # Disable inputs if model failed to train
        if not model_trained:
            self.hours_entry.configure(state="disabled")
            self.attendance_entry.configure(state="disabled")
            self.predict_button.configure(state="disabled")
            self.result_label.configure(text="Model not trained. Check console for errors.", text_color="red")


    def predict_score(self):
        """
        Retrieves user input, validates it, and uses the trained model
        to predict the final score, updating the GUI with the result.
        """
        hours_input = self.hours_entry.get()
        attendance_input = self.attendance_entry.get()

        if not hours_input or not attendance_input:
            self.result_label.configure(text="Please fill in all fields.", text_color="orange")
            return

        try:
            # Convert inputs to float
            new_hrs = float(hours_input)
            new_attendance = float(attendance_input)

            # Prediction
            predicted_score = model.predict([[new_hrs, new_attendance]])
            
            # Display the result
            self.result_label.configure(text=f"Predicted Score: {predicted_score[0]:.2f}", text_color="white")

        except ValueError:
            # Handle cases where input is not a valid number
            self.result_label.configure(text="Invalid input. Please enter numbers only.", text_color="red")
        except Exception as e:
            # Handle other potential errors during prediction
            self.result_label.configure(text=f"An error occurred: {e}", text_color="red")


if __name__ == "__main__":
    # Set appearance mode and color theme
    ctk.set_appearance_mode("System")  # Modes: "System" (default), "Dark", "Light"
    ctk.set_default_color_theme("blue")  # Themes: "blue" (default), "green", "dark-blue"
    
    app = App()
    app.mainloop()
