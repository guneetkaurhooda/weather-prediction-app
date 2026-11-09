import pandas as pd
import matplotlib.pyplot as plt

# Load processed weather data
data = pd.read_csv("data/mumbai_weather_processed.csv")

# Display first few rows
print("\nFirst 5 rows:")
print(data.head())

# Display dataset shape
print("\nDataset shape:")
print(data.shape)

# Display data types
print("\nData types:")
print(data.dtypes)

# Display summary statistics
print("\nSummary statistics:")
print(data.describe())

# Display target distribution
print("\nRain Tomorrow distribution:")
print(data["Rain_Tomorrow"].value_counts())

# Plot target distribution
data["Rain_Tomorrow"].value_counts().plot(kind="bar")

plt.title("Rain Tomorrow Distribution")
plt.xlabel("Rain Tomorrow")
plt.ylabel("Number of Days")
plt.tight_layout()
plt.show()

# Weather feature distributions
features = [
    "temp",
    "tmin",
    "tmax",
    "rhum",
    "wspd",
    "pres"
]

for feature in features:
    plt.figure()
    plt.hist(data[feature], bins=20)
    plt.title(f"Distribution of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    # Compare average weather conditions for Rain vs No Rain
print("\nAverage weather conditions by Rain_Tomorrow:")
print(data.groupby("Rain_Tomorrow")[features].mean())

# Compare weather features for Rain vs No Rain
for feature in features:
    plt.figure()
    data.boxplot(column=feature, by="Rain_Tomorrow")
    plt.title(f"{feature} vs Rain Tomorrow")
    plt.suptitle("")
    plt.xlabel("Rain Tomorrow")
    plt.ylabel(feature)
    plt.tight_layout()
    plt.show()

    # Correlation between weather features
print("\nCorrelation matrix:")
print(data[features].corr())