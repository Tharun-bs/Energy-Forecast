import pandas as pd
import numpy as np

def generate_synthetic_data(usage_type='domestic', days=366):
    """
    Generate synthetic data for energy consumption, machine load, and other variables for either
    'domestic' or 'industrial' use over a given number of days.
    """
    # Define ranges based on usage type
    if usage_type == 'domestic':
        energy_range = (10, 50)  # kWh per day
        machine_load_range = (0.1, 2)  # kW
        temperature_range = (-10, 40)  # Celsius
        humidity_range = (20, 80)  # %
        solar_radiation_range = (100, 250)  # W/m²
    elif usage_type == 'industrial':
        energy_range = (500, 10000)  # kWh per day
        machine_load_range = (5, 500)  # kW
        temperature_range = (10, 50)  # Celsius
        humidity_range = (30, 90)  # %
        solar_radiation_range = (150, 400)  # W/m²

    # Generate random data for each column
    data = {
        'Timestamp': pd.date_range('2024-01-01', periods=days, freq='D'),
        'Energy_Consumption_kWh': np.random.uniform(energy_range[0], energy_range[1], days),
        'Machine_Load_kW': np.random.uniform(machine_load_range[0], machine_load_range[1], days),
        'Temperature_C': np.random.uniform(temperature_range[0], temperature_range[1], days),
        'Humidity_%': np.random.uniform(humidity_range[0], humidity_range[1], days),
        'Solar_Radiation_W_m2': np.random.uniform(solar_radiation_range[0], solar_radiation_range[1], days),
        'Time_of_Day': np.random.choice(['Morning', 'Noon', 'Evening', 'Night'], days),
        'Day_of_Week': pd.date_range('2024-01-01', periods=days, freq='D').dayofweek,
        'Is_Holiday': np.random.choice([0, 1], days)  # 0 for regular day, 1 for holiday
    }

    # Create the DataFrame
    df = pd.DataFrame(data)

    # Calculate Real-Time Power Load (kWh/kW)
    df['Real-Time Power Load (kWh/kW)'] = df['Energy_Consumption_kWh'] / df['Machine_Load_kW']
    df['Real-Time Power Load (kWh/kW)'] = df['Real-Time Power Load (kWh/kW)'].replace([np.inf, -np.inf], np.nan)  # Replace infinities with NaN

    return df

# Generate synthetic domestic data
domestic_data = generate_synthetic_data(usage_type='domestic')

# Generate synthetic industrial data
industrial_data = generate_synthetic_data(usage_type='industrial')

# Example: Save the generated synthetic data to CSV
domestic_data.to_csv('synthetic_domestic_energy_data.csv', index=False)
industrial_data.to_csv('synthetic_industrial_energy_data.csv', index=False)

print("Synthetic data generated and saved as CSV files.")
