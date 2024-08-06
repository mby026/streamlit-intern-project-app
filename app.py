# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import streamlit as st
from google.oauth2 import service_account
from google.cloud import bigquery

# Authenticate using Streamlit secrets
try:
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"]
    )
except Exception as e:
    st.error(f"Error loading Google Cloud credentials: {e}")
    st.stop()

# Set your project ID
project_id = 'q-chang-data-sandbox'

# Initialize a BigQuery client
try:
    client = bigquery.Client(credentials=credentials, project=project_id)
except Exception as e:
    st.error(f"Error initializing BigQuery client: {e}")
    st.stop()

# Read the CSV file
csv_file_path = '/content/drive/MyDrive/bq-results-20240711-070338-1720681439559/bq-results-20240711-070338-1720681439559.csv'
try:
    df_job_2023 = pd.read_csv(csv_file_path)
except FileNotFoundError:
    st.error(f"CSV file not found: {csv_file_path}")
    st.stop()
except Exception as e:
    st.error(f"Error reading CSV file: {e}")
    st.stop()

# Your SQL query
query_job_2024 = """
SELECT *
FROM `q-chang-data-sandbox.indv_arkaporn.download_q_chang_operation_t`
"""

# Execute the query and convert the result to a pandas DataFrame
try:
    df_job_2024 = client.query(query_job_2024).to_dataframe()
except Exception as e:
    st.error(f"Error executing BigQuery query: {e}")
    st.stop()

# Convert 'created_date' to datetime and format to show only the date part
df_job_2023['created_date'] = pd.to_datetime(df_job_2023['created_date']).dt.date

# Sort the dataframe by 'created_date' in ascending order
df_job_2023 = df_job_2023.sort_values(by='created_date')

# Convert 'created_date' to datetime
df_job_2024['created_date'] = pd.to_datetime(df_job_2024['created_date'])

# Sort the dataframe by 'created_date' in ascending order
df_job_2024 = df_job_2024.sort_values(by='created_date')

# Concatenate the two DataFrames
df_job = pd.concat([df_job_2023, df_job_2024])

# Optionally reset the index
df_job = df_job.reset_index(drop=True)

# Convert 'created_date' to datetime
df_job['created_date'] = pd.to_datetime(df_job['created_date'])

# Streamlit App
st.title('LSTM Forecasting Dashboard')
st.write('This dashboard allows you to view and filter forecasted unique job code counts.')

# Sidebar for filtering
st.sidebar.header('Filter options')
sale_model_options = st.sidebar.multiselect('Select Sale Model', df_job['sale_model'].unique(), default=df_job['sale_model'].unique())
channel_options = st.sidebar.multiselect('Select Channel', df_job['channel'].unique(), default=df_job['channel'].unique())
type_ow_name_options = st.sidebar.multiselect('Select Type of Work Name', df_job['type_ow_name'].unique(), default=df_job['type_ow_name'].unique())
type_of_job_options = st.sidebar.multiselect('Select Type of Job', df_job['type_of_job'].unique(), default[df_job['type_of_job'].unique()])
team_name_options = st.sidebar.multiselect('Select Team Name', df_job['team_name'].unique(), default[df_job['team_name'].unique()])
grade_options = st.sidebar.multiselect('Select Grade', df_job['grade'].unique(), default[df_job['grade'].unique()])
sub_district_options = st.sidebar.multiselect('Select Sub District', df_job['address_info_sub_district_name'].unique(), default[df_job['address_info_sub_district_name'].unique()])
district_options = st.sidebar.multiselect('Select District', df_job['address_info_district_name'].unique(), default[df_job['address_info_district_name'].unique()])
province_options = st.sidebar.multiselect('Select Province', df_job['address_info_province_name'].unique(), default[df_job['address_info_province_name'].unique()])

# Filter the dataframe based on the selected options
filtered_df = df_job[
    (df_job['sale_model'].isin(sale_model_options)) &
    (df_job['channel'].isin(channel_options)) &
    (df_job['type_ow_name'].isin(type_ow_name_options)) &
    (df_job['type_of_job'].isin(type_of_job_options)) &
    (df_job['team_name'].isin(team_name_options)) &
    (df_job['grade'].isin(grade_options)) &
    (df_job['address_info_sub_district_name'].isin(sub_district_options)) &
    (df_job['address_info_district_name'].isin(district_options)) &
    (df_job['address_info_province_name'].isin(province_options))
]

# Group by 'created_date' and count unique 'job_code'
filtered_job_count = filtered_df.groupby(filtered_df['created_date'])['job_code'].nunique().reset_index()
filtered_job_count.columns = ['created_date', 'unique_job_code_count']

# Rename columns for consistency
filtered_job_count = filtered_job_count.rename(columns={'created_date': 'ds', 'unique_job_code_count': 'y'})

# Drop rows with NaN values
filtered_job_count = filtered_job_count.dropna()

# Ensure the 'ds' column is in datetime format and set as index
filtered_job_count['ds'] = pd.to_datetime(filtered_job_count['ds'])
filtered_job_count.set_index('ds', inplace=True)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(filtered_job_count[['y']])

# Create sequences for LSTM
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 10
X, y = create_sequences(scaled_data, seq_length)

# Reshape input to be [samples, time steps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))

# Build the LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
lstm_model.add(LSTM(units=50))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
lstm_model.fit(X, y, epochs=20, batch_size=32)

# Forecast future values (e.g., for the next 30 days)
future_steps = 30
future_input = scaled_data[-seq_length:]
future_forecast = []

for _ in range(future_steps):
    future_pred = lstm_model.predict(future_input.reshape(1, seq_length, 1))
    future_forecast.append(future_pred[0, 0])
    future_input = np.append(future_input[1:], future_pred)

future_forecast = scaler.inverse_transform(np.array(future_forecast).reshape(-1, 1))

# Create a DataFrame for future forecast
future_dates = pd.date_range(start=filtered_job_count.index[-1], periods=future_steps+1, freq='D')[1:]
forecast_df = pd.DataFrame(data=future_forecast, index=future_dates, columns=['y'])

# Combine historical data with forecast
combined_df = pd.concat([filtered_job_count, forecast_df], axis=0)

# Date range filter for the combined data
start_date = st.date_input('Start date', combined_df.index.min())
end_date = st.date_input('End date', combined_df.index.max())

if start_date > end_date:
    st.error('Error: End date must fall after start date.')
else:
    filtered_combined_df = combined_df[start_date:end_date]

    st.write(f"Forecasted data from {start_date} to {end_date}")
    st.line_chart(filtered_combined_df)

    # Display filtered forecast data
    st.write(filtered_combined_df)
