# a class and methods to save daily AQI data to a database
from models import City, AQIData, AQIForecast, ModelFiles, Base
import os
import pandas as pd
import requests
from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

database_url = os.environ['DATABASE_URL']
waqi_api_key = os.environ['WAQI_API_KEY']


class AQIOperator:
    def __init__(self):
        # Create a SQLAlchemy engine that is connected to the database
        self.engine = create_engine(
            database_url
        )

        # Create a Session object
        Session = sessionmaker(bind=self.engine)
        self.session = Session()

        # If tables don't exist, create them
        Base.metadata.create_all(self.engine)

        # get list of cities
        self.cities = pd.read_csv('cities.csv')

    def daily_data(self, data):
        # Extract the data for the new day and add it to the dictionary
        # This is needed because the keys from the API can change from day to day
        day_data = {}

        day_data['aqi'] = data['aqi']
        day_data['dominentpol'] = data['dominentpol']
        day_data['datetime'] = data['time']['s']

        if 'no2' in data['iaqi']:
            day_data['no2'] = data['iaqi']['no2']['v']
        else:
            day_data['no2'] = 0
        if 'o3' in data['iaqi']:
            day_data['o3'] = data['iaqi']['o3']['v']
        else:
            day_data['o3'] = 0
        if 'pm25' in data['iaqi']:
            day_data['pm25'] = data['iaqi']['pm25']['v']
        else:
            day_data['pm25'] = 0
        if 'dew' in data['iaqi']:
            day_data['dew'] = data['iaqi']['dew']['v']
        else:
            day_data['dew'] = 0
        if 'h' in data['iaqi']:
            day_data['h'] = data['iaqi']['h']['v']
        else:
            day_data['h'] = 0
        if 'p' in data['iaqi']:
            day_data['p'] = data['iaqi']['p']['v']
        else:
            day_data['p'] = 0
        if 'pm10' in data['iaqi']:
            day_data['pm10'] = data['iaqi']['pm10']['v']
        else:
            day_data['pm10'] = 0
        if 'so2' in data['iaqi']:
            day_data['so2'] = data['iaqi']['so2']['v']
        else:
            day_data['so2'] = 0
        if 't' in data['iaqi']:
            day_data['t'] = data['iaqi']['t']['v']
        else:
            day_data['t'] = 0
        if 'w' in data['iaqi']:
            day_data['w'] = data['iaqi']['w']['v']
        else:
            day_data['w'] = 0
        if 'wg' in data['iaqi']:
            day_data['wg'] = data['iaqi']['wg']['v']
        else:
            day_data['wg'] = 0
        if 'co' in data['iaqi']:
            day_data['co'] = data['iaqi']['co']['v']
        else:
            day_data['co'] = 0
        day_data['timestamp_local'] = data['time']['s']

        # Return the resulting dictionary
        return day_data

    def save_aqi_data(self):
        # loop through cities
        for index, row in self.cities.iterrows():
            # get aqi data for city
            url = "https://api.waqi.info/feed/geo:{};{}/?token={}".format(
                row['lat'], row['lon'], waqi_api_key)
            response = requests.get(url)
            # create city object if it doesn't exist
            city = self.session.query(City).filter(
                City.city_name == row['city_name']).first()
            if not city:
                city = City(
                    city_name=row['city_name'], country_code=row['country_code'], lat=row['lat'], lon=row['lon'])
                # add city object to session
                self.session.add(city)
                # commit changes to database
                self.session.commit()
            # create aqi data object if it doesn't exist
            aqi_data = self.session.query(AQIData).filter(
                AQIData.city_id == city.id, AQIData.timestamp_local == datetime.fromisoformat(response.json()['data']['time']['s'])).first()
            if not aqi_data:
                new_daily_data = self.daily_data(response.json()['data'])
                aqi_data = AQIData(city_id=city.id, aqi=new_daily_data['aqi'], co=new_daily_data['co'], dew=new_daily_data['dew'], h=new_daily_data['h'], no2=new_daily_data['no2'], o3=new_daily_data['o3'], p=new_daily_data['p'], pm10=new_daily_data[
                                   'pm10'], pm25=new_daily_data['pm25'], so2=new_daily_data['so2'], t=new_daily_data['t'], w=new_daily_data['w'], wg=new_daily_data['wg'], timestamp_local=datetime.fromisoformat(new_daily_data['timestamp_local']))
                # add aqi data object to session
                self.session.add(aqi_data)
                # commit changes to database
                self.session.commit()

        # Close the session
        self.session.close()
        print("Done")

    def get_aqi_data_all(self):
        # get all aqi data
        aqi_data = self.session.query(AQIData).all()
        # Close the session
        self.session.close()
        return aqi_data

    def get_all_cities(self):
        # get all cities
        cities = self.session.query(City).all()
        # Close the session
        self.session.close()
        return cities

    def get_aqi_data_by_city(self, city_name):
        # get aqi data for city
        city = self.session.query(City).filter(
            City.city_name == city_name).first()
        if city:
            aqi_data = self.session.query(AQIData).filter(
                AQIData.city_id == city.id).all()
            if aqi_data:
                city.aqi_data = aqi_data
        # Close the session
        self.session.close()
        return city

    def get_aqi_data_by_lat_lon(self, lat, lon):
        # get aqi data for city
        city = self.session.query(City).filter(
            City.lat == lat, City.lon == lon).first()
        if city:
            aqi_data = self.session.query(AQIData).filter(
                AQIData.city_id == city.id).all()
            if aqi_data:
                city.aqi_data = aqi_data
        # Close the session
        self.session.close()
        return city

    def get_aqi_forecast_by_city(self, city_name):
        # get aqi data for city
        city = self.session.query(City).filter(
            City.city_name == city_name).first()
        if city:
            aqi_forecast = self.session.query(AQIForecast).filter(
                AQIForecast.city_id == city.id).all()
            if aqi_forecast:
                city.aqi_forecast = aqi_forecast
        # Close the session
        self.session.close()
        return city

    def save_aqi_forecast_by_city(self, city_name, forecast_data):
        # get aqi data for city
        city = self.session.query(City).filter(
            City.city_name == city_name).first()
        if city:
            for data in forecast_data:
                aqi_forecast = self.session.query(AQIForecast).filter(
                    AQIForecast.city_id == city.id, AQIForecast.timestamp_local == datetime.fromisoformat(data['timestamp_local'])).first()
                if not aqi_forecast:
                    aqi_forecast = AQIForecast(
                        city_id=city.id, aqi=data['aqi'], timestamp_local=datetime.fromisoformat(data['timestamp_local']))
                    # add aqi data object to session
                    self.session.add(aqi_forecast)
                    # commit changes to database
                    self.session.commit()
        # Close the session
        self.session.close()

    def get_model_file_by_city(self, city_name):
        # get model file for city
        city = self.session.query(City).filter(
            City.city_name == city_name).first()
        if city:
            model_files = self.session.query(ModelFiles).filter(
                ModelFiles.city_id == city.id).all()
            if model_files:
                city.model_file = model_files
        # Close the session
        self.session.close()
        return city

    def save_model_file_by_city(self, city_name, model_file):
        # save model file for city
        city = self.session.query(City).filter(
            City.city_name == city_name).first()
        if city:
            model_file = ModelFiles(
                city_id=city.id, file=model_file, file_name=model_file, file_date=datetime.now())
            # add model file object to session
            self.session.add(model_file)
            # commit changes to database
            self.session.commit()
        # Close the session
        self.session.close()
