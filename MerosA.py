import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

# Load data
data = pd.read_csv('weather_data.csv')

# Calculate statistics
numeric_cols = ['value']
stats = data[numeric_cols].describe()
print("Statistics of Numeric Columns:\n", stats)

# Categorical data
categorical_cols = ['identifier','value_type_id','location_id','source_id']
for col in categorical_cols:
    print(f"Frequency of {col}:\n", data[col].value_counts())

# Encode categorical data
encoder = OneHotEncoder(sparse_output=False)  # Use sparse_output
encoded_categorical = encoder.fit_transform(data[categorical_cols])
encoded_df = pd.DataFrame(encoded_categorical, columns=encoder.get_feature_names_out(categorical_cols))

# Convert to Unix Time Stamp
data['timestamp'] = pd.to_datetime(data['timestamp'])
data['timestamp'] = data['timestamp'].astype('int64') // 10**9  

# Scale
scaler = MinMaxScaler()
data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

# Drop unnecessary columns
data = data.drop(columns=['id','identifier','value_type_id','location_id','source_id'])

# Concatenate with encoded data
data = pd.concat([data, encoded_df], axis=1)

# Save preprocessed DataFrame to CSV file
data.to_csv('preprocessed_weather_data.csv', index=False)
print("Csv file created\n")