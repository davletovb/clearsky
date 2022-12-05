from aqioperator import AQIOperator
from aqiforecaster import ARIMAForecaster
import pandas as pd
import json


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


def get_forecast(city_name, forecast_length):
    """
    Train ARIMA model and get AQI forecast for the next n days.
    Args:
        city_name: Name of the city.
    Returns:
        Aqi forecast for the next n days.
    """
    # get historical aqi data
    data = get_history_data(city_name)
    # get aqi forecast
    model_params = {
        'order': (1, 1, 1)
    }

    forecaster = ARIMAForecaster(model_params)

    data = forecaster.prepare_data(data)

    forecaster.fit(data)

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

    aqi_operator = AQIOperator()
    aqi_operator.save_model_file_by_city(
        city_name, model_name='ARIMA', model_parameters=model_params)

    aqi_operator.save_aqi_forecast_by_city(
        city_name, forecasts[0], model_name='ARIMA')

    return forecasts


def save_daily_forecasts():
    """
    Save daily forecasts for all cities. 
    This function is called by a cron job.
    """
    aqi_operator = AQIOperator()
    cities = aqi_operator.get_cities()
    for row, city in cities.iterrows():
        get_forecast(city['city_name'], 1)


if __name__ == '__main__':
    save_daily_forecasts()