import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# Load Dataset from a CSV File
file_path = "rec_crop.csv"  # Replace with your CSV file path
df = pd.read_csv(file_path)

# Map categorical data to numeric values for the model
season_mapping = {season: i for i, season in enumerate(df['Season'].unique())}
weather_mapping = {weather: i for i, weather in enumerate(df['Weather'].unique())}
month_mapping = {month: i for i, month in enumerate(df['Month'].unique())}
crop_mapping = {crop: i for i, crop in enumerate(df['Crop'].unique())}
reverse_crop_mapping = {v: k for k, v in crop_mapping.items()}

df['Season'] = df['Season'].map(season_mapping)
df['Weather'] = df['Weather'].map(weather_mapping)
df['Month'] = df['Month'].map(month_mapping)
df['Crop'] = df['Crop'].map(crop_mapping)

# Features (X) and Target (y)
X = df[['Season', 'Weather', 'Month']]
y = df['Crop']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Predict and Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Function to Recommend Crop (using season, weather, and month)
def recommend_crop(season, weather, month):
    season_num = season_mapping.get(season, None)
    weather_num = weather_mapping.get(weather, None)
    month_num = month_mapping.get(month, None)
    
    if season_num is None or weather_num is None or month_num is None:
        return "Invalid input. Please ensure season, weather, and month are correct."
    
    crop_num = model.predict([[season_num, weather_num, month_num]])[0]
    return reverse_crop_mapping[crop_num]

# GUI Setup
def on_recommend():
    season = season_var.get()
    weather = weather_var.get()
    month = month_var.get()
    
    recommended_crop = recommend_crop(season, weather, month)
    
    # Display the result in a message box
    messagebox.showinfo("Recommended Crop", f"Recommended Crop for {season.capitalize()} with {weather.capitalize()} weather in {month.capitalize()} is: {recommended_crop}")

# Create the main window
root = tk.Tk()
root.title("Crop Recommendation System")


# Set the window size
root.geometry("600x400")  # Increase window size for better spacing

# Increase font size for better visibility
font = ('Arial', 14)

# Create and place the widgets
tk.Label(root, text="Select the Season:", font=font).grid(row=0, column=0, padx=10, pady=20, sticky='w')
season_var = tk.StringVar(root)
season_var.set("Summer")  # Default value
season_menu = tk.OptionMenu(root, season_var, "Summer", "Winter", "Rainy")
season_menu.config(font=font, width=20)
season_menu.grid(row=0, column=1, padx=10, pady=20)

tk.Label(root, text="Select the Weather:", font=font).grid(row=1, column=0, padx=10, pady=20, sticky='w')
weather_var = tk.StringVar(root)
weather_var.set("Hot")  # Default value
weather_menu = tk.OptionMenu(root, weather_var, "Hot", "Cold", "Humid")
weather_menu.config(font=font, width=20)
weather_menu.grid(row=1, column=1, padx=10, pady=20)

tk.Label(root, text="Select the Month:", font=font).grid(row=2, column=0, padx=10, pady=20, sticky='w')
month_var = tk.StringVar(root)
month_var.set("January")  # Default value
month_menu = tk.OptionMenu(root, month_var, 
                           "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December")
month_menu.config(font=font, width=20)
month_menu.grid(row=2, column=1, padx=10, pady=20)

# Button to recommend the crop with larger size
recommend_button = tk.Button(root, text="Recommend Crop", font=font, command=on_recommend, width=20)
recommend_button.grid(row=3, column=0, columnspan=2, pady=30)

# Run the GUI event loop
root.mainloop()
