#💻 Energy-Forecast
This project is a full-stack machine learning application designed to forecast energy consumption for both domestic and industrial settings using Facebook’s Prophet time series model. It uses a Flask backend, HTML/CSS frontend, and synthetic data generated for training. 
# ⚡ Energy Consumption Forecasting App

A web-based application that predicts **domestic** or **industrial** energy consumption using Facebook Prophet, Flask, and synthetic datasets.

![screenshot](screenshot.png) <!-- Add an actual screenshot of the app UI if available -->

## 🌟 Features

- Predicts daily energy consumption (kWh) based on user inputs.
- Handles both **Domestic** and **Industrial** usage types.
- Models trained using synthetic data via Prophet.
- Beautiful animated and responsive frontend using HTML/CSS.
- Easy-to-use date selector and categorical input form.

---

## 🚀 How It Works

1. **Synthetic Data Generation**  
   The `data.py` script generates synthetic energy consumption data for both domestic and industrial sectors.

2. **Model Training**  
   `df_energy.py` loads the generated CSVs, processes them, and trains two separate Prophet models which are serialized using `pickle`.

3. **Web Interface**  
   The `app.py` Flask app serves a web interface (`index.html`) that allows users to input variables like temperature, time of day, humidity, etc., and receive an energy forecast.

---

## 📦 Project Structure

```bash
.
├── app.py                              # Flask backend
├── data.py                             # Synthetic data generator
├── df_energy.py                        # Training and saving Prophet models
├── index.html                          # Frontend HTML with animations
├── synthetic_domestic_energy_data.csv  # Synthetic data for domestic use
├── synthetic_industrial_energy_data.csv (optional if included)
├── synthetic_industrial_energy_data_trained_prophet_model.pkl
├── synthetic_domestic_energy_data_trained_prophet_model.pkl

