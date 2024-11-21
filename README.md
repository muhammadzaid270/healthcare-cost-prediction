# Healthcare Cost Prediction App

This app predicts annual healthcare insurance costs based on user inputs like age, sex, BMI, number of children, smoking status, and region using a trained Random Forest model.

## Features
- Predicts healthcare costs based on user input.
- Provides an explanation of how each factor affects the cost.
- Easy-to-use GUI built with **Tkinter**.

## Requirements
- Python 3.x
- Libraries: `pandas`, `numpy`, `scikit-learn`, `tkinter`

Install required libraries with:
```bash
pip install pandas numpy scikit-learn
```
## Running the Program
1. Clone this repository or download source code.
2. Install the dependencies.
3. Place the `medical_cost.csv` dataset in the same directory.
4. Run the script:
   ```bash
   python healthcare_cost_prediction.py
   ```
5. Enter details in the GUI and click "Predict" to get the healthcare cost estimate.

## How It Works
- **Data**: Preprocessed with `LabelEncoder` for categorical data.
- **Model**: Trained using `RandomForestRegressor` to predict healthcare costs.
- **GUI**: Inputs are entered via Tkinter, and results are displayed in a new window.

## License
MIT License