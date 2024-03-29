# Streamlit dashboard for AQI forecasting, using various models and parameters.
import streamlit as st
import pandas as pd
import plotly.express as px
import datetime

from aqiforecaster import ARIMAForecaster
from aqiforecaster import XGBoostForecaster
from aqiforecaster import ExponentialSmoothingForecaster
from aqioperator import AQIOperator
from aqi_interpreter import AQIInterpreter


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
    data = aqi_operator.get_aqi_dataframe(city_name)

    return data


def get_daily_forecasted_data(city_name, forecast_model='ARIMA'):
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
    # sort data by timestamp
    data.sort_values(by='timestamp_local', inplace=True)

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
            'order': (1,  0,  1)
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
        model_params: Parameters for the forecast model.
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
        forecast_length: Number of days for which AQI forecast is required.
        forecaster: Name of the forecast model.
        model_params: Parameters for the forecast model.
    Returns:
        Aqi forecast for the next n days.
    """
    # get historical aqi data
    data = get_history_data(city_name)

    # get aqi forecast
    forecaster = get_forecast_model(forecaster, model_params)

    forecaster = forecaster.fit(data)

    forecasts = forecaster.predict(forecast_length)

    return forecasts


def get_interpretation(city_name, todays_data, history_data, future_forecasts, model_name='ARIMA'):
    """
    Get interpretation for the AQI forecast.
    Args:
        city_name: Name of the city.
        todays_data: AQI data for today.
        history_data: Historical AQI data.
        future_forecasts: AQI forecast for the next n days.
    Returns:
        Interpretation for the AQI forecast.
    """

    # check if the interpretation for the forecast already exists in the database
    aqi_operator = AQIOperator()

    date = datetime.datetime.now().strftime('%Y-%m-%d')

    interpretation = aqi_operator.get_aqi_interpretation_by_city(
        city_name, date, model_name)

    if interpretation:
        return interpretation

    # generate interpretation for the forecast if it does not exist in the database
    interpreter = AQIInterpreter()

    interpretation = interpreter.interpret(
        city_name, todays_data, history_data, future_forecasts)

    # save interpretation to database
    aqi_operator.save_aqi_interpretation_by_city(
        city_name, interpretation, date, model_name)

    return interpretation


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
    history_data = get_history_data(city_name)

    # get daily forecasted data
    history_forecasts = get_daily_forecasted_data(city_name)

    # get aqi forecast
    future_forecasts = get_forecast(city_name, forecast_length,
                                    forecaster, model_params)

    # get interpretation for the forecast
    interpretation = get_interpretation(
        city_name, history_data.iloc[-1], history_data, future_forecasts, forecaster)

    # convert forecasts to dataframe with timestamp_local and aqi as column names
    future_forecasts = pd.DataFrame(future_forecasts)
    future_forecasts['timestamp_local'] = pd.to_datetime(
        future_forecasts['timestamp_local'])

    st.markdown('### Air Quality Index Forecast')
    st.markdown(
        'Here is the historical and forecasted air quality index for the next {} days in {}'.format(forecast_length, city_name))
    st.markdown('---')

    st.markdown('#### Explanation')
    st.markdown(interpretation)

    # plot historical aqi data and forecasted aqi data on the same chart, with different colors
    fig = px.line(history_data, x='timestamp_local', y='aqi',
                  title='Historical AQI Data', labels={'timestamp_local': 'Date', 'aqi': 'AQI'}, height=500, width=800)
    fig.add_scatter(x=history_forecasts['timestamp_local'], y=history_forecasts['aqi'],
                    mode='lines', name='Forecasted AQI Data', line=dict(color='red'))
    fig.add_scatter(x=future_forecasts['timestamp_local'], y=future_forecasts['aqi'],
                    mode='markers', name='Future Forecasted AQI Data', line=dict(color='red'))
    st.plotly_chart(fig)

    # plot forecasted aqi data
    fig = px.line(future_forecasts, x='timestamp_local', y='aqi',
                  title='Future Forecasted AQI Data', labels={'timestamp_local': 'Date', 'aqi': 'AQI'})
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

    # show a sidebar with the app title and description
    st.sidebar.title('ClearSky')
    st.sidebar.markdown(
        'This is a demo app for forecasting air quality index. The data is from [The World AirQuality Project](https://aqicn.org/).')
    st.sidebar.markdown('---')
    # the forecast model options
    st.sidebar.markdown('## Forecasting Options')

    city_name = st.sidebar.selectbox('City', cities)

    forecast_length = st.sidebar.slider('Forecast Length', 1, 7, 3)

    forecaster = st.sidebar.selectbox('Forecast Model', forecast_model_names)
    st.sidebar.markdown('Model Parameters')
    forecast_model_default_params = get_default_params(forecaster)

    # show forecast model parameters based on the selected forecast model and allow user to change them
    # show each element of the dictionary as a text input box with the key as the label
    model_params = {}
    for key, value in forecast_model_default_params.items():
        # check the type of the value
        if isinstance(value, int):
            model_params[key] = st.sidebar.number_input(key, value)
        elif isinstance(value, float):
            model_params[key] = st.sidebar.number_input(key, value)
        elif isinstance(value, str):
            model_params[key] = st.sidebar.text_input(key, value)
        elif isinstance(value, bool):
            model_params[key] = st.sidebar.checkbox(key, value)
        elif isinstance(value, list):
            # if the value is a list, show each element of the list as a text input box
            list_params = {}
            for i, v in enumerate(value):
                list_params[i] = st.sidebar.text_input(key + ' ' + str(i), v)
            model_params[key] = list_params
        elif isinstance(value, dict):
            # if the value is a dictionary, show each element of the dictionary as a text input box
            dict_params = {}
            for k, v in value.items():
                dict_params[k] = st.sidebar.text_input(key + ' ' + k, v)
            model_params[key] = dict_params
        elif isinstance(value, tuple):
            # if the value is a tuple, show each element of the tuple as a text input box
            tuple_params = []
            for i, v in enumerate(value):
                if isinstance(v, int):
                    tuple_params.append(st.sidebar.number_input(
                        label=key + ' ' + str(i), value=v, min_value=0))
                elif isinstance(v, float):
                    tuple_params.append(st.sidebar.number_input(
                        label=key + ' ' + str(i), value=v, min_value=0.0))
                elif isinstance(v, str):
                    tuple_params.append(
                        st.sidebar.text_input(key + ' ' + str(i), v))
            tuple_params = tuple(tuple_params)
            model_params[key] = tuple_params

        else:
            model_params[key] = st.sidebar.text_input(key, value)

    model_params = {k: v for k, v in model_params.items() if v != ''}

    forecast_model_default_params.update(model_params)

    if st.sidebar.button('Show Forecast'):
        show_data_and_forecasts(city_name, forecast_length,
                                forecaster, forecast_model_default_params)


if __name__ == '__main__':
    main()
