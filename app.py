from flask import Flask, render_template, request, jsonify
from prophet import Prophet
import pickle
import pandas as pd
from datetime import datetime

# Initialize the Flask app
app = Flask(__name__)

# Load both the trained Prophet models (one for domestic and one for industrial)
with open(r'C:\Users\Tharun B S\Documents\VS Code\energy_forecast\synthetic_domestic_energy_data_trained_prophet_model.pkl', 'rb') as f:
    domestic_model = pickle.load(f)

with open(r'C:\Users\Tharun B S\Documents\VS Code\energy_forecast\synthetic_industrial_energy_data_trained_prophet_model.pkl', 'rb') as f:
    industrial_model = pickle.load(f)

# Load both synthetic datasets
synthetic_domestic_data = pd.read_csv(r"C:\Users\Tharun B S\Documents\VS Code\energy_forecast\synthetic_domestic_energy_data.csv")
synthetic_industrial_data = pd.read_csv(r"C:\Users\Tharun B S\Documents\VS Code\energy_forecast\synthetic_industrial_energy_data.csv")

# Mapping categorical input values to numeric values
temperature_map = {"Cold": 1, "Moderate": 2, "Hot": 3}
humidity_map = {"Low": 1, "Moderate": 2, "High": 3}
solar_radiation_map = {"Low": 1, "Moderate": 2, "High": 3}
time_of_day_map = {"Morning": 1, "Noon": 2, "Evening": 3, "Night": 4}
holiday_indicator_map = {"0": 0, "1": 1}  # Weekday = 0, Weekend/Holiday = 1
appliance_usage_map = {"0": 0, "1": 1}  # Low = 0, High = 1
usage_type_map = {"Domestic": 1, "Industrial": 2}

# Route to display the home page (with input form)
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get all form inputs
        timestamp_str = request.form['timestamp']
        timestamp = datetime.strptime(timestamp_str, '%d-%m-%Y')  # Fixed format to DD-MM-YYYY

        # Get other inputs and map categorical values to numeric
        temperature = temperature_map[request.form['temperature']]
        humidity = humidity_map[request.form['humidity']]
        solar_radiation = solar_radiation_map[request.form['solar_radiation']]
        time_of_day = time_of_day_map[request.form['time_of_day']]
        holiday_indicator = holiday_indicator_map[request.form['holiday_indicator']]
        appliance_usage = appliance_usage_map[request.form['appliance_usage']]
        usage_type = request.form['usage_type']

        # Select the correct dataset and model based on usage type (Domestic or Industrial)
        if usage_type == "Domestic":
            data = synthetic_domestic_data
            model = domestic_model
        else:
            data = synthetic_industrial_data
            model = industrial_model

        # Create a DataFrame for Prophet with the timestamp as input
        df = pd.DataFrame({
            'ds': [timestamp],
            'y': [0]  # Placeholder for energy consumption; Prophet will predict the 'y' value
        })

        # Add more data from form inputs
        df['temperature'] = temperature
        df['humidity'] = humidity
        df['solar_radiation'] = solar_radiation
        df['time_of_day'] = time_of_day
        df['holiday_indicator'] = holiday_indicator
        df['appliance_usage'] = appliance_usage
        df['usage_type'] = usage_type

        # Use the selected model to make a prediction
        forecast = model.predict(df)
        predicted_energy = forecast['yhat'][0]  # Predicted energy consumption value

        # Format the timestamp to show as DD-MM-YYYY (without HH:MM:SS)
        formatted_timestamp = timestamp.strftime('%d-%m-%Y')

        # Return the prediction along with the form input to the HTML template
        return render_template('index.html', 
                               predicted_energy=predicted_energy, 
                               timestamp=formatted_timestamp,  # Display formatted date
                               temperature=temperature,
                               humidity=humidity,
                               solar_radiation=solar_radiation,
                               time_of_day=time_of_day,
                               holiday_indicator=holiday_indicator,
                               appliance_usage=appliance_usage,
                               usage_type=usage_type)

    except Exception as e:
        # Return the error message if an exception occurs
        return render_template('index.html', error=str(e))


if __name__ == '__main__':
    app.run(debug=True)
