import pandas as pd

# Load raw weather data
data = pd.read_csv(
    "data/mumbai_weather_raw.csv",
    index_col="time",
    parse_dates=True
)

# Select useful weather features
features = [
    "temp",
    "tmin",
    "tmax",
    "rhum",
    "wspd",
    "pres"
]

data = data[features + ["prcp"]]

# Create a separate table containing tomorrow's rainfall
tomorrow_rain = data[["prcp"]].copy()
tomorrow_rain.index = tomorrow_rain.index - pd.Timedelta(days=1)
tomorrow_rain = tomorrow_rain.rename(columns={"prcp": "tomorrow_prcp"})

# Match today's weather with the actual next calendar day's rainfall
data = data.join(tomorrow_rain)

# Create the prediction target
data["Rain_Tomorrow"] = data["tomorrow_prcp"].apply(
    lambda x: "Yes" if x > 0 else "No"
)

# Remove rows where tomorrow's rainfall is unknown
data = data.dropna(subset=["tomorrow_prcp"])

# Remove rows with missing feature values
data = data.dropna(subset=features)

# Keep only features and target
data = data[features + ["Rain_Tomorrow"]]

# Display processed data
print("\nProcessed data:")
print(data.head())

print("\nDataset shape:")
print(data.shape)

print("\nMissing values:")
print(data.isna().sum())

print("\nTarget distribution:")
print(data["Rain_Tomorrow"].value_counts())

# Save processed dataset
data.to_csv("data/mumbai_weather_processed.csv")

print("\nProcessed dataset saved successfully.")