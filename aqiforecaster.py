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

        return X

    def fit(self, X):
        """Fit ARIMA model to training data.

        Args:
            X: Training data input.
            y: Training data output.

        Returns:
            Self.
        """
        from statsmodels.tsa.arima.model import ARIMA as ARIMA_model
        self.model = ARIMA_model(X, **self.params)
        self.model = self.model.fit()
        return self

    def predict(self, X, n_periods):
        """
        Predict output for given input data for n periods.

        Args:
            X: Input data.
            n_periods: Number of periods to predict.

        Returns:
            Predicted output.
        """
        predictions = self.model.predict(
            start=len(X), end=len(X)+n_periods-1)

        return predictions


class XGBoostForecaster(TimeSeriesForecaster):
    """Wrapper for XGBoost model."""

    def __init__(self, params):
        """Initialize XGBoost model with given parameters.

        Args:
            params: Dictionary of parameters for XGBoost model.
        """
        super().__init__(params)
        self.model = None

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

        # Resample the data to a daily frequency
        data = data.resample('D').mean()

        # Split the train and test samples 80%:20%
        train_size = int(len(data) * 0.8)
        train, test = data[0:train_size], data[train_size:len(data)]

        # Drop missing values
        train = train.dropna()
        test = test.dropna()

        return train, test

    def fit(self, X, y):
        """Fit XGBoost model to training data.

        Args:
            X: Training data input.
            y: Training data output.

        Returns:
            Self.
        """
        from xgboost import XGBRegressor
        self.model = XGBRegressor(**self.params)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """Predict output for given input data.

        Args:
            X: Input data.

        Returns:
            Predicted output.
        """
        return self.model.predict(X)


class ExponentialSmoothingForecaster(TimeSeriesForecaster):
    """Wrapper for exponential smoothing model."""

    def __init__(self, params):
        """Initialize exponential smoothing model with given parameters.

        Args:
            params: Dictionary of parameters for exponential smoothing model.
        """
        super().__init__(params)
        self.model = None

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

        return X

    def fit(self, X):
        """Fit exponential smoothing model to training data.

        Args:
            X: Training data input.
            y: Training data output.

        Returns:
            Self.
        """
        from statsmodels.tsa.holtwinters import ExponentialSmoothing as ExponentialSmoothing_model
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
        # Predict the next 7 days
        predictions = self.model.predict(n)
        return predictions
