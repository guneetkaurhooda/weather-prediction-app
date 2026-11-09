import pandas as pd

# Load the processed dataset
data = pd.read_csv("data/mumbai_weather_processed.csv")

# Separate input features (X) and target (y)
features = ["temp", "tmin", "tmax", "rhum", "wspd", "pres"]

X = data[features]
y = data["Rain_Tomorrow"].map({"No": 0, "Yes": 1})

# Split chronologically: first 80% for training, remaining 20% for testing
split_index = int(len(data) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# Display dataset information
print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())