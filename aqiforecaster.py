from abc import ABC, abstractmethod


class TimeSeriesForecaster(ABC):
    """Abstract base class for time series forecasting."""

    def __init__(self, params):
        """Initialize time series forecaster with given parameters.

        Args:
            params: Dictionary of parameters for time series forecast model.
        """
        self.params = params

    @abstractmethod
    def prepare_data(self, data):
        """Prepare data for training and prediction.

        Args:
            data: Data to prepare.

        Returns:
            Prepared data.
        """
        pass

    @abstractmethod
    def fit(self, X, y):
        """Fit time series forecast model to training data.

        Args:
            X: Training data input.
            y: Training data output.

        Returns:
            Self.
        """
        pass

    @abstractmethod
    def predict(self, X):
        """Predict output for given input data.

        Args:
            X: Input data.

        Returns:
            Predicted output.
        """
        pass


class ARIMAForecaster(TimeSeriesForecaster):
    """Wrapper for ARIMA model."""

    def __init__(self, params):
        """Initialize ARIMA model with given parameters.

        Args:
            params: Dictionary of parameters for ARIMA model.
        """
        super().__init__(params)
        self.model = None
        self.X = None

    def prepare_data(self, data):
        """Prepare data for training and prediction.

        Args:
            data: Data to prepare.

        Returns:
            Prepared data.
        """
        # Drop NaN values
        data.dropna(inplace=True)

        # Set the index to timestamp column if it is not already
        if data.index.name != 'timestamp_local':
            data.set_index('timestamp_local', inplace=True)

        # resample data to daily frequency
        data = data.resample('D').mean()

        # Return the AQI values, ARIMA model does not need the other columns
        X = data['aqi']

        self.X = X

        return X

    def fit(self, data):
        """Fit ARIMA model to training data.

        Args:
            X: Training data input.
            y: Training data output.

        Returns:
            Self.
        """
        from statsmodels.tsa.arima.model import ARIMA as ARIMA_model
        X = self.prepare_data(data)
        self.model = ARIMA_model(X, **self.params)
        self.model = self.model.fit()
        return self

    def predict(self, n_periods):
        """
        Predict output for given input data for n periods.

        Args:
            X: Input data.
            n_periods: Number of periods to predict.

        Returns:
            Predicted output.
        """
        import json

        forecasts = self.model.predict(
            start=len(self.X), end=len(self.X)+n_periods-1)

        # convert forecasts to json with timestamp_local and aqi as column names
        forecasts = forecasts.reset_index()
        forecasts = forecasts.rename(
            columns={'index': 'timestamp_local', 'predicted_mean': 'aqi', 0: 'aqi'})
        forecasts['timestamp_local'] = forecasts['timestamp_local'].dt.strftime(
            '%Y-%m-%d')

        # convert forecasts to json
        forecasts = json.loads(forecasts.to_json(orient='records'))

        return forecasts


class XGBoostForecaster(TimeSeriesForecaster):
    """Wrapper for XGBoost model."""

    def __init__(self, params):
        """Initialize XGBoost model with given parameters.

        Args:
            params: Dictionary of parameters for XGBoost model.
        """
        super().__init__(params)
        self.model = None
        self.X = None
        self.y = None
        self.last_date = None

    def prepare_data(self, data):
        """Prepare data for training and prediction.

        Args:
            data: Data to prepare.

        Returns:
            Prepared data.
        """
        import pandas as pd
        # Drop NaN values
        data.dropna(inplace=True)

        # Set the index to timestamp column if it is not already
        # if data.index.name != 'timestamp_local':
        #    data.set_index('timestamp_local', inplace=True)

        # Resample the data to a daily frequency
        #data = data.resample('D').mean()

        X = data.drop('aqi', axis=1)
        # convert timestamp_local to integer
        self.last_date = X['timestamp_local'].iloc[-1]
        X['timestamp_local'] = pd.to_numeric(X['timestamp_local'])
        y = data['aqi']

        self.X = X
        self.y = y

        return X, y

    def fit(self, data):
        """Fit XGBoost model to training data.

        Args:
            X: Training data input.
            y: Training data output.

        Returns:
            Self.
        """
        from xgboost import XGBRegressor
        import xgboost as xgb

        self.model = XGBRegressor(**self.params)

        #X, y, _, _ = self.prepare_data(data)
        X, y = self.prepare_data(data)

        # X = xgb.DMatrix(data=data.drop('aqi', axis=1), label=data['aqi'], feature_names=data.drop(
        #    'aqi', axis=1).columns, enable_categorical=True)

        self.model.fit(X, y)

        return self

    def predict(self, n_periods):
        """Predict output for given input data.

        Args:
            X: Input data.
            n_periods: Number of periods to predict.

        Returns:
            Predicted output.
        """
        import datetime
        import pandas as pd

        # create a list of dates from the last date in the dataset to the next n days
        future_dates = [
            self.last_date + datetime.timedelta(days=x) for x in range(0, n_periods)]

        future_data = pd.DataFrame()

        # add the dates to the dataframe
        future_data['timestamp_local'] = [date for date in future_dates]
        future_data.sort_values(
            'timestamp_local', inplace=True, ascending=True)
        # copy the timestamp_local column to a new list to be used later
        future_timestamps = future_data['timestamp_local'].copy()
        future_data['timestamp_local'] = pd.to_numeric(
            future_data['timestamp_local'])

        forecasts = self.model.predict(future_data)

        # convert forecasts to json with timestamp_local and aqi as column names
        forecasts = pd.DataFrame(forecasts)
        forecasts = forecasts.rename(
            columns={0: 'aqi'})
        forecasts['timestamp_local'] = future_timestamps

        return forecasts


class ExponentialSmoothingForecaster(TimeSeriesForecaster):
    """Wrapper for exponential smoothing model."""

    def __init__(self, params):
        """Initialize exponential smoothing model with given parameters.

        Args:
            params: Dictionary of parameters for exponential smoothing model.
        """
        super().__init__(params)
        self.model = None
        self.X = None

    def prepare_data(self, data):
        """Prepare data for training and prediction.

        Args:
            data: Data to prepare.

        Returns:
            Prepared data.
        """
        # Drop NaN values
        data.dropna(inplace=True)

        # Set the index to timestamp column if it is not already
        if data.index.name != 'timestamp_local':
            data.set_index('timestamp_local', inplace=True)

        # resample data to daily frequency
        data = data.resample('D').mean()

        X = data['aqi']

        self.X = X

        return X

    def fit(self, data):
        """Fit exponential smoothing model to training data.

        Args:
            X: Training data input.
            y: Training data output.

        Returns:
            Self.
        """
        from statsmodels.tsa.holtwinters import ExponentialSmoothing as ExponentialSmoothing_model
        X = self.prepare_data(data)
        self.model = ExponentialSmoothing_model(
            X, **self.params)
        self.model = self.model.fit()
        return self

    def predict(self, n):
        """Predict output for given input data.

        Args:
            X: Input data.

        Returns:
            Predicted output.
        """
        import json

        # Predict the next n days
        forecasts = self.model.predict(len(self.X), len(self.X)+n-1)

        # convert forecasts to json with timestamp_local and aqi as column names
        forecasts = forecasts.reset_index()
        forecasts = forecasts.rename(
            columns={'index': 'timestamp_local', 'predicted_mean': 'aqi', 0: 'aqi'})
        forecasts['timestamp_local'] = forecasts['timestamp_local'].dt.strftime(
            '%Y-%m-%d')

        # convert forecasts to json
        forecasts = json.loads(forecasts.to_json(orient='records'))

        return forecasts
