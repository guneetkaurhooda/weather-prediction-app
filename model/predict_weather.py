import joblib
import pandas as pd

# Load the saved model
model = joblib.load("model/weather_model.pkl")

print("Mumbai Weather Prediction")
print("-------------------------")

# Take weather inputs from the user
temp = float(input("Enter temperature: "))
tmin = float(input("Enter minimum temperature: "))
tmax = float(input("Enter maximum temperature: "))
rhum = float(input("Enter humidity: "))
wspd = float(input("Enter wind speed: "))
pres = float(input("Enter atmospheric pressure: "))

# Prepare input data
input_data = pd.DataFrame([{
    "temp": temp,
    "tmin": tmin,
    "tmax": tmax,
    "rhum": rhum,
    "wspd": wspd,
    "pres": pres
}])

# Make prediction
prediction = model.predict(input_data)[0]

# Display result
if prediction == 1:
    print("\nPrediction: It may rain tomorrow!")
else:
    print("\nPrediction: No rain expected tomorrow.")

# Display prediction probabilities
probabilities = model.predict_proba(input_data)[0]

print("\nPrediction Confidence:")
for class_label, probability in zip(model.classes_, probabilities):
    label = "Rain" if class_label == 1 else "No Rain"
    print(label, ":", round(probability * 100, 2), "%")