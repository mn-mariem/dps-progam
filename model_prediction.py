import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
import joblib
import pickle
from sklearn.linear_model import LinearRegression 


# Import Data
df = pd.read_csv('monatszahlen2307_verkehrsunfaelle_10_07_23_nosum.csv')

# Remove null values 
df.dropna(inplace=True)

# Select the most relevent columns
df = df[['MONATSZAHL', 'AUSPRAEGUNG', 'JAHR', 'MONAT', 'WERT']]

# Convert the 'MONAT' to a datetime format
df['MONAT'] = pd.to_datetime(df['MONAT'], format='%Y%m', errors='coerce')

# Extract year and month as separate columns
df['year'] = df['MONAT'].dt.year
df['month'] = df['MONAT'].dt.month

# Drop 'MONAT' and  'JAHR'  columns
df.drop(['MONAT', 'JAHR'], axis=1, inplace=True)

# Split the data into training set and testing set
train = df[df.year <= 2020]
test = df[df.year > 2020]

# Prepare traing and testing vectors
target = 'WERT'
x_train = train.drop(target, axis=1)
y_train = train[target]
x_test = test.drop(target, axis=1)
y_test = test[target]

# Define columns to one-hot encode
categorical_columns = ['AUSPRAEGUNG', 'MONATSZAHL']

# Define one hot encoder
encoder = OneHotEncoder(handle_unknown='error')

# Apply one-hot encoding to the specified columns on the training set
# x_train_encoded = pd.DataFrame(encoder.fit_transform(x_train[categorical_columns]), columns=encoder.get_feature_names_out(categorical_columns))
transformed_data  = encoder.fit_transform(x_train[categorical_columns]).toarray()

# Get feature names with input_features parameter set
feature_names_out = encoder.get_feature_names_out(input_features=categorical_columns)

# the above transformed_data is an array so convert it to dataframe
encoded_data = pd.DataFrame(transformed_data, columns=feature_names_out, index=x_train.index)

# now concatenate the original data and the encoded data using pandas
x_train_final = pd.concat([x_train[['year', 'month']], encoded_data], axis=1)

# Convert 'Year' and 'Month' back to datetime if needed
x_train_final['year'] = pd.to_datetime(x_train_final['year'], format='%Y').dt.year
x_train_final['month'] = pd.to_datetime(x_train_final['month'], format='%m').dt.month

# Train the model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(x_train_final, y_train)


## Apply the same one-hot encoding transformation on the test set
transformed_data_test = encoder.transform(x_test[categorical_columns]).toarray()
print(x_test[categorical_columns][0:0].shape)

# Convert the transformed data array to a DataFrame for testing set
encoded_data_test = pd.DataFrame(transformed_data_test, columns=feature_names_out, index=x_test.index)

# Concatenate the original data and the encoded data using pandas for testing set
x_test_final = pd.concat([x_test[['year', 'month']], encoded_data_test], axis=1)

# Convert 'Year' and 'Month' back to datetime if needed
x_test_final['year'] = pd.to_datetime(x_test_final['year'], format='%Y').dt.year
x_test_final['month'] = pd.to_datetime(x_test_final['month'], format='%m').dt.month

# Make prediction
predictions = rf_model.predict(x_test_final)

# Evaluate the model
mae = mean_absolute_error(y_test, predictions)
print(f'Mean Absolute Error: {mae}')

# Calculate R2 score
r2 = r2_score(y_test, predictions)
print(f'R2 score: {r2}')

# # Save the model
joblib.dump(rf_model, 'rf_model.pkl')

# Save the encoder
pickle.dump(encoder, open('encoder.pickle', 'wb'))

