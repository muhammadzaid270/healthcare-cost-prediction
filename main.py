import tkinter as tk
from tkinter import messagebox
from tkinter import Toplevel
from tkinter import font as tkfont
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np

# Load and preprocess the data
df = pd.read_csv('medical_cost.csv')
df['region'] = df['region'].fillna('southwest')

# Encode categorical variables using Label Encoding
label_encoder_sex = LabelEncoder()
label_encoder_smoker = LabelEncoder()
label_encoder_region = LabelEncoder()

df['sex'] = label_encoder_sex.fit_transform(df['sex'])
df['smoker'] = label_encoder_smoker.fit_transform(df['smoker'])
df['region'] = label_encoder_region.fit_transform(df['region'])

X = df.drop(columns=['Id', 'charges'])
y = df['charges']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

cross_val_score_mean = np.mean(cross_val_score(model, X_scaled, y, cv=5))
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

# Tkinter App Class
class HealthcareCostApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Healthcare Cost Prediction")
        self.root.geometry("600x700")
        self.root.config(bg="#f7f8fc")
        
        # Center window on the screen
        self.center_window(self.root)
        
        # Heading Label
        self.heading_label = tk.Label(root, text="Healthcare Cost Prediction", font=("Arial", 18, 'bold'), bg="#f7f8fc", fg="#333")
        self.heading_label.pack(pady=20)
        
        # Form Section
        self.form_frame = tk.Frame(root, bg="#f7f8fc")
        self.form_frame.pack(pady=20)
        
        self.create_form(self.form_frame)
        
        # Predict Button
        self.predict_button = tk.Button(root, text="Predict", command=self.predict_cost, font=("Arial", 14), bg="#4CAF50", fg="white", relief="solid", width=20)
        self.predict_button.pack(pady=20)

    def center_window(self, window):
        # Get window width and height
        window_width = 600
        window_height = 500
        # Get screen width and height
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()
        # Calculate position x and y to center the window
        position_top = int(screen_height / 2 - window_height / 2)
        position_right = int(screen_width / 2 - window_width / 2)
        # Set the window's geometry
        window.geometry(f'{window_width}x{window_height}+{position_right}+{position_top}')

    def create_form(self, frame):
        # Age Input
        self.age_label = tk.Label(frame, text="Age:", font=("Arial", 12), bg="#f7f8fc", anchor="w")
        self.age_label.grid(row=0, column=0, pady=5, padx=20, sticky="w")
        self.age_input = tk.Entry(frame, font=("Arial", 12))
        self.age_input.grid(row=0, column=1, pady=5, padx=20, sticky="ew")
        
        # Sex Input
        self.sex_label = tk.Label(frame, text="Sex:", font=("Arial", 12), bg="#f7f8fc", anchor="w")
        self.sex_label.grid(row=1, column=0, pady=5, padx=20, sticky="w")
        self.sex_input = tk.StringVar(frame)
        self.sex_input.set("male")  # default value
        self.sex_dropdown = tk.OptionMenu(frame, self.sex_input, "male", "female")
        self.sex_dropdown.grid(row=1, column=1, pady=5, padx=20, sticky="ew")
        
        # BMI Input
        self.bmi_label = tk.Label(frame, text="BMI:", font=("Arial", 12), bg="#f7f8fc", anchor="w")
        self.bmi_label.grid(row=2, column=0, pady=5, padx=20, sticky="w")
        self.bmi_input = tk.Entry(frame, font=("Arial", 12))
        self.bmi_input.grid(row=2, column=1, pady=5, padx=20, sticky="ew")
        
        # Children Input
        self.children_label = tk.Label(frame, text="Children:", font=("Arial", 12), bg="#f7f8fc", anchor="w")
        self.children_label.grid(row=3, column=0, pady=5, padx=20, sticky="w")
        self.children_input = tk.Entry(frame, font=("Arial", 12))
        self.children_input.grid(row=3, column=1, pady=5, padx=20, sticky="ew")
        
        # Smoker Input
        self.smoker_label = tk.Label(frame, text="Smoker:", font=("Arial", 12), bg="#f7f8fc", anchor="w")
        self.smoker_label.grid(row=4, column=0, pady=5, padx=20, sticky="w")
        self.smoker_input = tk.StringVar(frame)
        self.smoker_input.set("no")  # default value
        self.smoker_dropdown = tk.OptionMenu(frame, self.smoker_input, "yes", "no")
        self.smoker_dropdown.grid(row=4, column=1, pady=5, padx=20, sticky="ew")
        
        # Region Input
        self.region_label = tk.Label(frame, text="Region:", font=("Arial", 12), bg="#f7f8fc", anchor="w")
        self.region_label.grid(row=5, column=0, pady=5, padx=20, sticky="w")
        self.region_input = tk.StringVar(frame)
        self.region_input.set("southwest")  # default value
        self.region_dropdown = tk.OptionMenu(frame, self.region_input, "southwest", "southeast", "northwest", "northeast")
        self.region_dropdown.grid(row=5, column=1, pady=5, padx=20, sticky="ew")

    def predict_cost(self):
        try:
            # Validate the inputs
            age = self.age_input.get()
            bmi = self.bmi_input.get()
            children = self.children_input.get()

            if not age.isdigit() or not bmi.replace('.', '', 1).isdigit() or not children.isdigit():
                messagebox.showerror("Input Error", "Please enter valid numeric values for age, BMI, and children.")
                return
            
            age = float(age)
            bmi = float(bmi)
            children = int(children)

            sex = self.sex_input.get()
            smoker = self.smoker_input.get()
            region = self.region_input.get()

            # Encode the inputs
            sex = label_encoder_sex.transform([sex])[0]
            smoker = label_encoder_smoker.transform([smoker])[0]
            region = label_encoder_region.transform([region])[0]

            # Prepare the input for prediction as a DataFrame
            input_data = pd.DataFrame([[age, sex, bmi, children, smoker, region]],
                                      columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region'])

            # Scale the input data
            input_data_scaled = scaler.transform(input_data)

            # Make the prediction
            predicted_charge = model.predict(input_data_scaled)[0]

            # Create the result window (Toplevel)
            result_window = Toplevel(self.root)
            result_window.title("Prediction Result")
            result_window.geometry("500x400")
            result_window.config(bg="#f7f8fc")
            
            # Center the result window
            self.center_window(result_window)

            result_label = tk.Label(result_window, text=f"Predicted Healthcare Charge (Annual): ${predicted_charge:.2f}",
                                    font=("Arial", 14, 'bold'), bg="#f7f8fc", pady=20, fg="#333")
            result_label.pack(padx=20, pady=5)

            explanation = self.generate_explanation(age, bmi, smoker)

            explanation_label = tk.Label(result_window, text=explanation, font=("Arial", 10), bg="#f7f8fc", justify="left", anchor="w")
            explanation_label.pack(padx=20, pady=10)

        except Exception as e:
            messagebox.showerror("Error", f"Error: {e}. Please ensure you enter the data correctly.")

    def generate_explanation(self, age, bmi, smoker):
        explanation = ""
        if age > 50:
            explanation += f"- Your age is above 50, which tends to increase healthcare costs.\n"
        elif age > 30:
            explanation += f"- Your age is between 30 and 50, which is typically associated with moderate healthcare costs.\n"
        else:
            explanation += f"- Your age is below 30, which typically results in lower healthcare costs.\n"
        if smoker == 1:
            explanation += f"- Smoking significantly increases healthcare costs.\n"
        if bmi > 30:
            explanation += f"- A higher BMI is associated with higher healthcare costs.\n"
        explanation += f"\nThis model's cross-validated R² score is: {cross_val_score_mean:.2f}\n"
        explanation += f"Mean Absolute Error (MAE) on the test set: ${mae:.2f}\n"
        explanation += "\nNote: This is an estimate of annual health insurance charges in the United States."
        explanation += "\nThe model factors in age, sex, smoking status, BMI, and region of residence."
        return explanation

# Running the application
if __name__ == "__main__":
    root = tk.Tk()
    app = HealthcareCostApp(root)
    root.mainloop()