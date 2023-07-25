import torch
import pandas as pd
from datetime import datetime
import math
from sklearn.preprocessing import MinMaxScaler
from torch import nn
import datetime
from xgboost import XGBRegressor
import numpy as np
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
    def __init__(self, params):
        """Initialize XGBoost model with given parameters.

        Args:
            params: Dictionary of parameters for XGBoost model.
        """
        super().__init__(params)
        self.n_lags = 3
        self.start_date = None
        self.last_observation = None
        self.X = None
        self.model = XGBRegressor(**params)

    def create_lagged_features(self, df, n_lags):
        """Create lagged features for time series data."""
        for i in range(1, n_lags + 1):
            df[f'lag_{i}'] = df['aqi'].shift(i)
        df.dropna(inplace=True)
        return df

    def prepare_data(self, data):
        """Prepare data for training and prediction."""
        data = self.create_lagged_features(data, self.n_lags)
        data['timestamp_local'] = (
            data['timestamp_local'] - data['timestamp_local'].min()) / pd.Timedelta(days=1)
        X = data.drop('aqi', axis=1)
        y = data['aqi'].values
        return X, y

    def fit(self, data):
        """Fit XGBoost model to training data."""
        X, y = self.prepare_data(data)
        self.last_observation = list(y[-self.n_lags:])
        self.model.fit(X.values, y)
        self.X = X  # Store the DataFrame
        return self

    def predict(self, n_periods):
        """Predict output for given number of periods."""
        future_data = [self.last_observation[i] for i in range(-self.n_lags, 0)]
        predictions = []

        # Define start and end dates for the prediction period
        if self.start_date is None:
            self.start_date = datetime.date.today()
        end_date = self.start_date + datetime.timedelta(days=n_periods-1)

        # Create a date range for the prediction period
        future_dates = pd.date_range(self.start_date, end_date)

        for date in future_dates:
            # Convert Timestamp to datetime.date
            date = date.to_pydatetime().date()

            # Create a DataFrame for this date
            x_df = pd.DataFrame(index=[date], columns=self.X.columns)

            # Fill in the lagged values
            for i in range(1, self.n_lags + 1):
                x_df[f'lag_{i}'] = future_data[-i]

            # Fill in the timestamp_local value
            x_df['timestamp_local'] = (date - self.start_date).days

            # Make the prediction
            pred = self.model.predict(x_df.values)[0]
            predictions.append(pred)

            # Append the prediction to future_data for use in future predictions
            future_data.append(pred)

        # Create DataFrame for predictions
        prediction_dates = [self.start_date + datetime.timedelta(days=i) for i in range(n_periods)]
        predictions_df = pd.DataFrame()
        predictions_df['timestamp_local'] = prediction_dates
        predictions_df['aqi'] = predictions

        return predictions_df


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


class LSTMForecaster(TimeSeriesForecaster):
    """Wrapper for LSTM model."""

    def __init__(self, params):
        """Initialize LSTM model with given parameters.

        Args:
            params: Dictionary of parameters for LSTM model.
        """
        super().__init__(params)
        self.model = None
        self.scaler = MinMaxScaler()
        self.max_date = None

    def prepare_data(self, df):
        """Prepare data for training and prediction.

        Args:
            df: DataFrame that includes 'time' and 'air_quality'.

        Returns:
            inout_seq: A list of tuples where each tuple represents a sequence of 'window_size' 
            days of air quality data and the air quality of the next day.
        """
        # convert time to ordinal, because most models can't deal with datetime objects
        df['timestamp_local'] = df['timestamp_local'].apply(
            lambda x: x.toordinal())

        # define inputs and output
        X = df[['timestamp_local']].values
        y = df['aqi'].values

        # scale data
        X = self.scaler.fit_transform(X)

        # save the maximum date
        self.max_date = df['timestamp_local'].max()

        # create sequences
        inout_seq = []
        L = len(X)
        for i in range(L-self.params['window_size']):
            train_seq = X[i:i+self.params['window_size']]
            train_label = y[i+self.params['window_size']                            :i+self.params['window_size']+1]
            inout_seq.append((torch.FloatTensor(train_seq),
                             torch.FloatTensor(train_label)))

        return inout_seq

    def fit(self, data):
        """Fit LSTM model to training data.

        Args:
            data: Training data.

        Returns:
            Self.
        """
        class LSTM(nn.Module):
            def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
                super().__init__()
                self.hidden_layer_size = hidden_layer_size
                self.lstm = nn.LSTM(input_size, hidden_layer_size)
                self.linear = nn.Linear(hidden_layer_size, output_size)
                self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                                    torch.zeros(1, 1, self.hidden_layer_size))

            def forward(self, input_seq):
                predictions = []
                for i in range(input_seq.shape[0]):
                    lstm_out, self.hidden_cell = self.lstm(
                        input_seq[i].view(1, 1, -1), self.hidden_cell)
                    prediction = self.linear(lstm_out.view(1, -1))
                    predictions.append(prediction)
                return torch.stack(predictions)

        train_inout_seq = self.prepare_data(data)

        self.model = LSTM()
        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        epochs = self.params['epochs']

        for i in range(epochs):
            for seq, labels in train_inout_seq:
                optimizer.zero_grad()
                self.model.hidden_cell = (torch.zeros(1, 1, self.model.hidden_layer_size),
                                          torch.zeros(1, 1, self.model.hidden_layer_size))
                y_pred = self.model(seq)
                single_loss = loss_function(y_pred, labels)
                single_loss.backward()
                optimizer.step()

        return self

    def predict(self, n_periods):
        """Predict output for given input data.

        Args:
            n_periods: Number of periods to predict.

        Returns:
            Predicted output.
        """
        # create future time sequence
        future_dates = np.arange(
            self.max_date + 1, self.max_date + n_periods + 1)

        # scale future_dates
        future_dates = self.scaler.transform(future_dates.reshape(-1, 1))

        # convert to torch tensor
        future_dates = torch.FloatTensor(future_dates)

        # make predictions for the future dates
        future_preds = self.model(future_dates)

        # inverse transform to get the actual air quality values
        actual_predictions = self.scaler.inverse_transform(
            future_preds.detach().numpy().reshape(-1, 1))

        # create DataFrame of predictions and corresponding dates
        df_preds = pd.DataFrame(data=actual_predictions, columns=['aqi'])

        # convert future_dates tensor back to numpy array
        future_dates_np = future_dates.detach().numpy()

        print(actual_predictions.shape)
        print(future_dates_np.shape)
        # inverse transform to get the actual dates
        actual_dates = self.scaler.inverse_transform(future_dates_np)

        # convert to datetime
        df_preds['timestamp_local'] = pd.Series(
            [datetime.fromordinal(math.floor(date)) for date in future_dates_np])

        # format the 'timestamp_local' column
        df_preds['timestamp_local'] = df_preds['timestamp_local'].dt.strftime(
            '%Y-%m-%d')

        # convert DataFrame to list of dictionaries
        forecasts = df_preds.to_dict('records')

        return forecasts
