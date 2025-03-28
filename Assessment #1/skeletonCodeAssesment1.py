import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta

# Load appointment data
df = pd.read_csv("appointments.csv")  # Contains scheduled_time, actual_time, doctor_id, patient_id

# Feature Engineering
df['delay'] = (pd.to_datetime(df['actual_time'], dayfirst=True) - pd.to_datetime(df['scheduled_time'], dayfirst=True)).dt.total_seconds() / 60
df['hour'] = pd.to_datetime(df['scheduled_time'], dayfirst=True).dt.hour
df['day_of_week'] = pd.to_datetime(df['scheduled_time'], dayfirst=True).dt.dayofweek

# Ensure no negative delays
df['delay'] = df['delay'].apply(lambda x: max(0, x))

# Define features and target variable
features = ['doctor_id', 'hour', 'day_of_week']
target = 'delay'

# Train AI Model
X = df[features]
y = df[target]

model = RandomForestRegressor()
model.fit(X, y)

# Predict delay for future appointments
def predict_wait_time(doctor_id, scheduled_time):
    hour = scheduled_time.hour
    day_of_week = scheduled_time.weekday()
    
    # Pass as DataFrame with column names
    input_data = pd.DataFrame([[doctor_id, hour, day_of_week]], columns=features)
    
    return model.predict(input_data)[0]  # Predicted delay in minutes

# Example usage
scheduled_time = datetime(2024, 3, 26, 18, 30)
predicted_delay = predict_wait_time(doctor_id=5, scheduled_time=scheduled_time)

# Adjusted appointment time
new_scheduled_time = scheduled_time + timedelta(minutes=predicted_delay)
print(f"Adjusted Slot: Doctor 5, New Scheduled Time: {new_scheduled_time}")


