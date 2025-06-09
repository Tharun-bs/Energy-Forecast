import pandas as pd
from prophet import Prophet
import pickle
import numpy as np

# Function to process data and train the Prophet model
def train_model(file_path):
    # Read the data from the CSV file
    data = pd.read_csv(file_path)

    # Debug: Print column names to check for any discrepancies
    print(f"Columns in {file_path}:\n", data.columns)

    # Check for missing values in important columns
    print(f"Missing values in columns:\n", data.isnull().sum())

    # Check for any inf or NaN values in the 'Real-Time Power Load (kWh/kW)' column
    if 'Real-Time Power Load (kWh/kW)' not in data.columns:
        # Ensure the 'Energy_Consumption_kWh' and 'Machine_Load_kW' columns are available
        if 'Energy_Consumption_kWh' in data.columns and 'Machine_Load_kW' in data.columns:
            # Calculate the 'Real-Time Power Load (kWh/kW)' column
            data['Real-Time Power Load (kWh/kW)'] = data['Energy_Consumption_kWh'] / data['Machine_Load_kW']
            data['Real-Time Power Load (kWh/kW)'] = data['Real-Time Power Load (kWh/kW)'].replace([np.inf, -np.inf], np.nan)  # Replace infinities with NaN
        else:
            raise ValueError("Columns 'Energy_Consumption_kWh' and 'Machine_Load_kW' are required to calculate 'Real-Time Power Load (kWh/kW)'.")

    # Convert 'Timestamp' to datetime and ignore the time of day
    data['Timestamp'] = pd.to_datetime(data['Timestamp'], errors='coerce')

    # Drop rows where 'Timestamp' is invalid (NaT) or 'Real-Time Power Load (kWh/kW)' is NaN
    data = data.dropna(subset=['Timestamp', 'Real-Time Power Load (kWh/kW)'])

    # Ensure valid timestamps (convert to date only)
    data['Timestamp'] = data['Timestamp'].dt.date

    # Debug: Print the number of rows before and after dropping NaN values
    print(f"Rows before dropping NaN values: {len(data)}")
    print(f"Rows after dropping NaN values: {len(data)}")

    # Check if there are enough rows for Prophet
    if len(data) < 2:
        raise ValueError("Not enough valid rows for training. Ensure that the data contains valid timestamps and non-NaN energy consumption values.")
    
    # Prepare the data for Prophet (timestamp and real-time load)
    df_train_prophet = data[['Timestamp', 'Real-Time Power Load (kWh/kW)']].rename(
        columns={'Timestamp': 'ds', 'Real-Time Power Load (kWh/kW)': 'y'}
    )

    # Initialize and train the Prophet model
    model = Prophet()  # You can add hyperparameters here if you want
    model.fit(df_train_prophet)

    # Save the trained model as a pickle file
    model_filename = file_path.split("\\")[-1].replace(".csv", "_trained_prophet_model.pkl")
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)

    print(f"Trained model saved as {model_filename}")


# Define the paths for the domestic and industrial energy consumption data files
domestic_file_path = r"C:\Users\Tharun B S\Documents\VS Code\energy_forecast\synthetic_domestic_energy_data.csv"
industrial_file_path = r"C:\Users\Tharun B S\Documents\VS Code\energy_forecast\synthetic_industrial_energy_data.csv"

# Train model for domestic data
train_model(domestic_file_path)

# Train model for industrial data
train_model(industrial_file_path)
