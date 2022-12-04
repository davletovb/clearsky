from sqlalchemy import Column, Integer, String, Float, DateTime, BINARY, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class City(Base):
    __tablename__ = 'cities'

    id = Column(Integer, primary_key=True)
    city_name = Column(String)
    country_code = Column(String)
    lat = Column(Float)
    lon = Column(Float)

    aqi_data = relationship('AQIData')
    aqi_forecast = relationship('AQIForecast')
    model_files = relationship('ModelFiles')


class AQIData(Base):
    __tablename__ = 'aqi_data'

    id = Column(Integer, primary_key=True)
    city_id = Column(Integer, ForeignKey('cities.id'))
    aqi = Column(Float)
    co = Column(Float)
    dew = Column(Float)
    h = Column(Float)
    no2 = Column(Float)
    o3 = Column(Float)
    p = Column(Float)
    pm10 = Column(Float)
    pm25 = Column(Float)
    so2 = Column(Float)
    t = Column(Float)
    w = Column(Float)
    wg = Column(Float)
    timestamp_local = Column(DateTime)


class AQIForecast(Base):
    __tablename__ = 'aqi_forecast'

    id = Column(Integer, primary_key=True)
    city_id = Column(Integer, ForeignKey('cities.id'))
    aqi = Column(Float)
    timestamp_local = Column(DateTime)


class ModelFiles(Base):
    __tablename__ = 'model_files'

    id = Column(Integer, primary_key=True)
    city_id = Column(Integer, ForeignKey('cities.id'))
    file = Column(BINARY)
    file_name = Column(String)
    file_date = Column(DateTime)
