from aqioperator import AQIOperator


def save_daily_data():
    """
    Save daily aqi data to database.
    This function is called by a cron job every day at 13:00 PM EST.
    """
    aqi_operator = AQIOperator()
    aqi_operator.save_aqi_data()


if __name__ == "__main__":
    save_daily_data()
