import streamlit as st
from datetime import datetime, timedelta
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import pandas as pd

TODAY = datetime.today().strftime("%Y-%m-%d")


st.title("Stock Prediction App")
date_options = [(datetime.today() - timedelta(days=365 * i)).strftime("%Y-%m-%d") for i in range(1, 6)]
START = st.selectbox("Select a start date", date_options)

stocks = ("AAPL", "GOOG", "MSFT", "GME", "AMZN", "TSLA", "BTC-USD", "ETH-USD")
selected_stock = st.selectbox("Select company for prediction", stocks)

n_weeks = st.slider("Weeks of forcasting:", 1, 104)
period = n_weeks * 7


def load_data(ticker):
    # Download data from yfinance
    data = yf.download(ticker, START, TODAY, auto_adjust=False)

    # Reset the index
    data.reset_index(inplace=True)

    # Clean up column headers to remove the second header line
    if isinstance(data.columns, pd.MultiIndex):  # Check if columns are a MultiIndex
        data.columns = data.columns.get_level_values(0)  # Keep only the first level of the header

    return data

data_load_state = st.text("Load data...")
data = load_data(selected_stock)
data_load_state.text("Loading data... done!" )
st.write("Data after cleaning headers:", data.head())  # Debugging statement to check cleaned data

# Plot raw data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)


plot_raw_data()

# Prepare the data for Prophet
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

# Initialize and fit the Prophet model
model = Prophet()
model.fit(df_train)

# Create a future dataframe for predictions
future = model.make_future_dataframe(periods=period)

# Make predictions
forecast = model.predict(future)
st.subheader("Forecast data")
st.write(forecast.tail())

# Plot forecast
st.subheader("Forecast plot")
fig1 = plot_plotly(model, forecast)
st.plotly_chart(fig1)

# Plot components
st.subheader("Forecast components")
fig2 = model.plot_components(forecast)
st.write(fig2)

