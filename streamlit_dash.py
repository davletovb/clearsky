# Streamlit dashboard for AQI forecasting, using various models and parameters.

import streamlit as st
import pandas as pd
import plotly.express as px
import json

from aqiforecaster import ARIMAForecaster
from aqiforecaster import XGBoostForecaster
from aqiforecaster import ExponentialSmoothingForecaster
from aqioperator import AQIOperator

# import data
def get_city_list():
    """
    Get list of cities for which AQI data is available.

    Returns:
        List of cities.
    """

    aqi_operator = AQIOperator()
    cities = aqi_operator.get_cities()
    # convert cities dataframe to key value pairs for dropdown list, where key is city name and value is city id
    cities = cities[['city_id', 'city_name']].set_index('city_name')[
        'city_id'].to_dict()

    return cities


def get_history_data(city_name):
    """
    Get historical aqi data from database.
    Args:
        city_name: Name of the city.
    Returns:
        Historical aqi data and timestamps for the city as a pandas dataframe.
    """
    aqi_operator = AQIOperator()
    data = aqi_operator.get_aqi_data_by_city(city_name)
    # convert aqi data to pandas dataframe
    data = pd.DataFrame([{'city_id': aqi.city_id, 'aqi': aqi.aqi, 'co': aqi.co, 'dew': aqi.dew, 'h': aqi.h, 'no2': aqi.no2, 'o3': aqi.o3, 'p': aqi.p,
                        'pm10': aqi.pm10, 'pm25': aqi.pm25, 'so2': aqi.so2, 't': aqi.t, 'w': aqi.w, 'wg': aqi.wg, 'timestamp_local': aqi.timestamp_local} for aqi in data.aqi_data])
    # convert timestamp column to datetime
    data['timestamp_local'] = pd.to_datetime(data['timestamp_local'])

    return data


def get_forecasted_data(city_name, forecast_model):
    """
    Get forecasted aqi data from database.
    Args:
        city_name: Name of the city.
        forecast_model: Name of the forecast model.
    Returns:
        Forecasted aqi data and timestamps for the city as a pandas dataframe.
    """
    aqi_operator = AQIOperator()
    data = aqi_operator.get_aqi_forecast_by_city(city_name, forecast_model)
    # convert aqi data to pandas dataframe
    data = pd.DataFrame([{'city_id': aqi.city_id, 'aqi': aqi.aqi,
                        'timestamp_local': aqi.timestamp_local} for aqi in data.aqi_forecasts])
    # convert timestamp column to datetime
    data['timestamp_local'] = pd.to_datetime(data['timestamp_local'])

    return data


def get_forecast_models():
    """
    Get list of available forecast models and corresponding class names.

    Returns:
        List of available forecast models.
    """

    forecast_models = [
        {'model_name': 'ARIMA', 'class_name': 'ARIMAForecaster'},
        {'model_name': 'XGBoost', 'class_name': 'XGBoostForecaster'},
        {'model_name': 'Exponential Smoothing',
            'class_name': 'ExponentialSmoothingForecaster'}
    ]

    return forecast_models


def get_forecast_model_params():
    """
    List of default parameters for each forecast model.

    """

    forecast_model_params = [
        {'model_name': 'ARIMA', 'default_params': {
            'order': (1, 0, 1)
        }},
        {'model_name': 'XGBoost', 'default_params': {
            'max_depth': 3, 'n_estimators': 100, 'learning_rate': 0.1}},
        {'model_name': 'Exponential Smoothing', 'default_params': {
            'trend': 'add', 'seasonal': 'add', 'seasonal_periods': 7}}
    ]

    return forecast_model_params


def get_forecast_model_class(forecast_model):
    """
    Get class name for a given forecast model.

    Args:
        forecast_model: Name of the forecast model.

    Returns:
        Class name for the forecast model.
    """

    forecast_models = get_forecast_models()

    for model in forecast_models:
        if model['model_name'] == forecast_model:
            return model['class_name']


def get_default_params(forecast_model):
    """
    Get default parameters for a given forecast model.

    Args:
        forecast_model: Name of the forecast model.

    Returns:
        Default parameters for the forecast model.
    """

    forecast_model_params = get_forecast_model_params()

    for model_params in forecast_model_params:
        # check if the model name matches the selected model
        if model_params['model_name'] == forecast_model:
            # access the default parameters for the model
            default_params = model_params['default_params']

    return default_params


def get_forecast_model(forecast_model, model_params):
    """
    Get class name for a given forecast model.

    Args:
        forecast_model: Name of the forecast model.

    Returns:
        Class name for the forecast model.
    """

    forecast_model_class = get_forecast_model_class(forecast_model)

    if forecast_model_class == 'ARIMAForecaster':
        return ARIMAForecaster(model_params)
    elif forecast_model_class == 'XGBoostForecaster':
        return XGBoostForecaster(model_params)
    elif forecast_model_class == 'ExponentialSmoothingForecaster':
        return ExponentialSmoothingForecaster(model_params)


def get_forecast(city_name, forecast_length, forecaster, model_params):
    """
    Train a forecast model and get AQI forecast for the next n days.
    Args:
        city_name: Name of the city.
    Returns:
        Aqi forecast for the next n days.
    """
    # get historical aqi data
    data = get_history_data(city_name)
    # get aqi forecast

    forecaster = get_forecast_model(forecaster, model_params)

    data = forecaster.prepare_data(data)

    forecaster = forecaster.fit(data)

    forecasts = forecaster.predict(data, forecast_length)

    # convert forecasts to json with timestamp_local and aqi as column names
    forecasts = forecasts.reset_index()
    forecasts = forecasts.rename(
        columns={'index': 'timestamp_local', 'predicted_mean': 'aqi', 0: 'aqi'})
    forecasts['timestamp_local'] = forecasts['timestamp_local'].dt.strftime(
        '%Y-%m-%d')

    # convert forecasts to json
    forecasts = json.loads(forecasts.to_json(orient='records'))
    # forecasts.to_json(orient='records')

    return forecasts


# show the data and forecasts on streamlit app
def show_data_and_forecasts(city_name, forecast_length, forecaster, model_params):
    """
    Show historical aqi data and forecasted aqi data on streamlit app.
    Args:
        city_name: Name of the city.
        forecast_length: Number of days to forecast.
        forecaster: Name of the forecaster.
        model_params: Parameters for the forecaster.
    Returns:
        None
    """
    # get historical aqi data
    data = get_history_data(city_name)
    # get aqi forecast
    forecasts = get_forecast(city_name, forecast_length,
                             forecaster, model_params)

    # convert forecasts to json with timestamp_local and aqi as column names
    forecasts = pd.DataFrame(forecasts)
    forecasts['timestamp_local'] = pd.to_datetime(forecasts['timestamp_local'])

    # plot historical aqi data
    fig = px.line(data, x='timestamp_local', y='aqi',
                  title='Historical AQI Data')
    st.plotly_chart(fig)

    # plot forecasted aqi data
    fig = px.line(forecasts, x='timestamp_local', y='aqi',
                  title='Forecasted AQI Data')
    st.plotly_chart(fig)


def main():
    """
    Main function.
    """

    # get list of available forecast models
    forecast_models = get_forecast_models()
    forecast_model_names = [model['model_name'] for model in forecast_models]

    # get list of available cities
    cities = get_city_list()

    # set page title
    st.title('Air Quality Index Forecasting')
    st.markdown('This is a demo app for forecasting air quality index.')
    st.markdown(
        'The data is from [The World AirQuality Project](https://aqicn.org/).')
    #
    st.markdown('---')
    st.markdown('## Forecasting')
    st.markdown('### Select a city')
    city_name = st.selectbox('City', cities)
    st.markdown('### Select a forecast model')
    forecaster = st.selectbox('Forecast Model', forecast_model_names)
    st.markdown('### Select forecast length')
    forecast_length = st.slider('Forecast Length', 1, 7, 3)
    st.markdown('### Select forecast model parameters')
    forecast_model_default_params = get_default_params(forecaster)
    model_params = st.selectbox('Forecast Model Parameters',
                                forecast_model_default_params)
    st.markdown('### Forecast')
    if st.button('Forecast'):
        show_data_and_forecasts(city_name, forecast_length,
                                forecaster, forecast_model_default_params)


if __name__ == '__main__':
    main()
