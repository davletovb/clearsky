# clearsky
The Air Quality Index forecasting for the big cities in Canada. The Air Quality Index, or AQI, is a measure of the air quality in a particular location. It is calculated using several factors, including the levels of pollutants in the air, such as ozone, particulate matter, carbon monoxide, and sulfur dioxide. The AQI is used to provide information to the public about the air quality in a given area and can help people make decisions about how to protect their health. It is typically reported on a scale from 0 to 500, with higher numbers indicating worse air quality.


To install these requirements, run this:

pip install -r requirements.txt

There are two parameters that needs to be set inside aqioperator.py

DATABASE_URL:
It can be set to 'sqlite:///db.db' to run locally. The project is currently using CockroachDB cloud, but it was tested with SQLite as well.

WAQI_API_KEY:
The data is from The World Air Quality Project. API key for data can be requested and obtained from https://aqicn.org/data-platform/token/ for free.
