import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# Load the dataset
df = pd.read_csv('/Users/divyanshjha/Developer/mutual_funds_audit/comprehensive_mutual_funds_data.csv')

# Replace '-' with NaN and handle conversion issues
df.replace('-', pd.NA, inplace=True)

# Convert columns to appropriate data types and handle missing values
numeric_features = ['min_sip', 'min_lumpsum', 'expense_ratio', 'fund_size_cr', 'fund_age_yr',
                    'sortino', 'alpha', 'sd', 'beta', 'sharpe']
for col in numeric_features:
    df[col] = pd.to_numeric(df[col], errors='coerce')  # Converts to numeric, making non-convertible values NaN

# Drop rows with missing values in critical numeric fields
df.dropna(subset=numeric_features, inplace=True)

# Normalize numerical features
scaler = MinMaxScaler()
df[numeric_features] = scaler.fit_transform(df[numeric_features])

# Encode categorical features
encoder = OneHotEncoder()
categorical_features = ['fund_manager', 'risk_level', 'amc_name', 'rating', 'category', 'sub_category']
encoded_features = encoder.fit_transform(df[categorical_features]).toarray()
encoded_feature_names = encoder.get_feature_names_out(categorical_features)
encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names)

# Combine all preprocessed features
final_df = pd.concat([df[numeric_features], encoded_df, df[['returns_1yr', 'returns_3yr', 'returns_5yr']]], axis=1)

# Define the target and features
X = final_df.drop(['returns_1yr', 'returns_3yr', 'returns_5yr'], axis=1)
y = final_df[['returns_1yr', 'returns_3yr', 'returns_5yr']]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape input to be [samples, time steps, features] for LSTM input
X_train = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(3))  # Output layer for predicting 1yr, 3yr, 5yr returns

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Predict and evaluate the model
predictions = model.predict(X_test)
print("Predictions:", predictions)

# Save the model in the recommended format
model.save('mutual_fund_return_model.keras')  # Save as a Keras file
