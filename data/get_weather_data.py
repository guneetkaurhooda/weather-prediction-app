from datetime import date
import meteostat as ms

#Mumbai weather station
station = "43003"

#Date range
start = date(2015,1,1)
end = date(2025,12,31)

#Get daily weather data
data = ms.daily(station,start,end)
data = data.fetch()

#Display the first few rows
print(data.head())

#Display available columns
print(data.columns)

#Display number of rows and columns
print(data.shape)

#Count missing values in each column
print(data.isna().sum())

#Display rainfall statistics
print("\nRainfall statistics:")
print(data["prcp"].describe())

#Count days with recorded rainfall
print("\nDays with rainfall: ")
print((data["prcp"]>0).sum())

#Count days with no rainfall
print("\nDays with no rainfall:")
print((data["prcp"]==0).sum())

#Check the data range of the dataset
print("\nFirst date:")
print(data.index.min())

print("\nLast date:")
print(data.index.max())

#Check how many dates are missing from the dataset 
all_dates = data.index.to_series().asfreq("D")
print("\nMissing dates:")
print(all_dates.isna().sum())