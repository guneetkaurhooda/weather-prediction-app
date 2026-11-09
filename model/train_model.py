import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load processed dataset
data = pd.read_csv("data/mumbai_weather_processed.csv")

# Select input features and target
features = ["temp", "tmin", "tmax", "rhum", "wspd", "pres"]

X = data[features]
y = data["Rain_Tomorrow"].map({"No": 0, "Yes": 1})

# Chronological train-test split
split_index = int(len(data) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# Create Gaussian Naive Bayes model
model = GaussianNB()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Classification report
print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["No Rain", "Rain"]
))

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("Model training completed!")
print("Accuracy:", accuracy * 100, "%")

# Save the trained model
joblib.dump(model, "model/weather_model.pkl")

print("Model saved successfully!")