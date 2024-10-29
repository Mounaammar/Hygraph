import numpy as np
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pandas as pd
import numpy as np
from fastdtw import fastdtw
import xarray as xr
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datetime import datetime

class TimeSeriesMetadata:
    def __init__(self, owner_id, element_type=''):
        self.owner_id = owner_id
        self.element_type = element_type

    def update_metadata(self,owner_id,element_id=''):
        self.owner_id=owner_id
        if element_id !='': self.element_type=element_id


class TimeSeries:
    """
     Create and add a multivariate time series to the graph.
     :param tsid: Time series ID
     :param timestamps: List of timestamps
     :param variables: List of variable names
     :param data: 2D array-like structure with data
     """
    def __init__(self, tsid, timestamps, variables, data, metadata=None):
        self.tsid = tsid
        time_index = pd.to_datetime(timestamps)
        self.data = xr.DataArray(data, coords=[time_index, variables], dims=['time', 'variable'], name=f'ts_{tsid}')
        self.metadata = metadata if metadata is not None else {}

    def append_data(self, date, value):
        """
            Append data to the time series, ensuring timestamps are unique and sequential.

            :param date: New timestamp for the data
            :param value: The value to append at this timestamp
            """
        # Convert the timestamp to nanosecond precision to match xarray's internal precision
        date = pd.Timestamp(date)
        if self.data.coords['time'].size > 0:
            last_timestamp = self.last_timestamp()
            if date <= last_timestamp:
                raise ValueError(f"New timestamp {date} must be after the last timestamp {last_timestamp}.")

        new_data = xr.DataArray([[value]], coords=[[date], self.data.coords['variable']], dims=['time', 'variable'])
        self.data = xr.concat([self.data, new_data], dim='time')

    def get_id(self):
        return

    def subset_time_series(self, start_time, end_time):
        """
        Return a subset of the time series between the given start and end times.

        :param start_time: Start time for the subset.
        :param end_time: End time for the subset.
        :return: A TimeSeries object with data between start_time and end_time.
        """
        # Ensure start_time and end_time are in pandas datetime format
        start_time = pd.to_datetime(start_time)
        end_time = pd.to_datetime(end_time)

        # Filter data within the specified time range
        subset_data = self.data.sel(time=slice(start_time, end_time))

        # Return a new TimeSeries object with the subset data
        return TimeSeries(
            tsid=f"{self.tsid}_subset",
            timestamps=subset_data.time.values,
            variables=self.data.coords['variable'].values,
            data=subset_data.values,
            metadata=self.metadata
        )
    def apply_aggregation(self, aggregation_name, start_time=None, end_time=None, variable_name=None):
        """
        Apply an aggregation function (like 'sum', 'mean', etc.) to the time series data,
        optionally filtered by a time range (start_time to end_time).

        :param aggregation_name: Aggregation function ('sum', 'mean', 'min', 'max', 'count', etc.)
        :param start_time: Start of the time range for filtering (optional).
        :param end_time: End of the time range for filtering (optional).
        :return: Aggregated value based on the specified function.
        """
        # Convert start_time and end_time to pandas datetime for filtering
        if start_time:
            start_time = pd.to_datetime(start_time)
        if end_time:
            end_time = pd.to_datetime(end_time)

        # Filter data within the given time range if specified
        if start_time or end_time:
            filtered_series = self.subset_time_series(start_time, end_time)
            filtered_data = filtered_series.data
        else:
            filtered_data = self.data
            # Check if filtered_data has any values
        if filtered_data.size == 0:
            return 0  # Return 0 if the array is empty
            # Select the specified variable if provided
        if variable_name:
            filtered_data = filtered_data.sel(variable=variable_name)
        # Apply aggregation on the filtered data

        if aggregation_name == 'sum':
            result= filtered_data.sum().values
        elif aggregation_name == 'mean':
            result = filtered_data.mean().values
        elif aggregation_name == 'min':
            result= filtered_data.min().values
        elif aggregation_name == 'max':
            result= filtered_data.max().values
        elif aggregation_name == 'count':
            result= len(filtered_data.coords['time'].values)
        else:
            raise ValueError(f"Unsupported aggregation: {aggregation_name}")
            # If result is a single element array, extract the scalar value
        return result.item() if np.isscalar(result) == False else result

    @staticmethod
    def aggregate_multiple(time_series_list, method='sum', fill_value=0):
        """
        Aggregates multiple multivariate time series using the specified method, aligning timestamps.

        :param time_series_list: List of TimeSeries objects to aggregate.
        :param method: Aggregation method ('sum', 'mean', 'min', 'max').
        :param fill_value: Value to fill for missing timestamps.
        :return: A new TimeSeries object with the aggregated result.
        """
        if not time_series_list:
            raise ValueError("No time series provided for aggregation.")

        # Collect all unique timestamps across all time series
        all_timestamps = sorted({timestamp for ts in time_series_list for timestamp in ts.data.time.values})

        # Define numpy aggregation functions for each method
        agg_funcs = {
            'sum': np.sum,
            'mean': np.mean,
            'min': np.min,
            'max': np.max
        }
        if method not in agg_funcs:
            raise ValueError(f"Unsupported aggregation method: {method}")

        # Determine variables from the first time series (assuming they all have the same variables)
        variables = time_series_list[0].data.coords['variable'].values

        # Initialize storage for aggregated data
        aggregated_data = []

        # Iterate over each timestamp and aggregate for each variable independently
        for timestamp in all_timestamps:
            timestamp_values = []

            for variable in variables:
                # Collect values for the current variable at this timestamp from all time series
                values_at_timestamp = []
                for ts in time_series_list:
                    if timestamp in ts.data.time.values:
                        val = ts.data.sel(time=timestamp, variable=variable).values
                        values_at_timestamp.append(val)
                    else:
                        values_at_timestamp.append(fill_value)

                # Apply the aggregation function to the collected values
                aggregated_value = agg_funcs[method](values_at_timestamp)
                timestamp_values.append(aggregated_value)

            aggregated_data.append(timestamp_values)

        # Convert aggregated data to xarray DataArray
        aggregated_data_array = xr.DataArray(
            data=aggregated_data,
            coords={'time': all_timestamps, 'variable': variables},
            dims=['time', 'variable']
        )

        # Return a new TimeSeries object with the aggregated data
        first_ts = time_series_list[0]
        return TimeSeries(tsid=f"{first_ts.tsid}_aggregated",
                          timestamps=all_timestamps,
                          variables=variables,
                          data=aggregated_data_array.values,
                          metadata=first_ts.metadata)

    def sum(self):
        """
        Return the sum of the time series data.
        """
        return np.sum(self.data.values)

    def mean(self):
        """
        Return the mean of the time series data.
        """
        return np.mean(self.data.values)

    def max(self):
        """
        Return the maximum value of the time series data.
        """
        return np.max(self.data.values)

    def min(self):
        """
        Return the minimum value of the time series data.
        """
        return np.min(self.data.values)

    def variance(self):
        """
        Return the variance of the time series data.
        """
        return np.var(self.data.values)

    def count(self):
        """
        Return the count of data points in the time series.
        """
        return self.data.size

        # --- Access Specific Values or Timestamps ---

    def get_value_at_timestamp(self, timestamp, variable_name=None):
        """
        Retrieve the value(s) at a specific timestamp for the given variable.

        :param timestamp: Timestamp for which the value is needed.
        :param variable_name: The name of the variable to retrieve (optional).
        :return: The value(s) at the given timestamp.
        """
        timestamp = pd.to_datetime(timestamp)
        try:
            if variable_name:
                return self.data.sel(time=timestamp)[variable_name].values
            return self.data.sel(time=timestamp).values
        except KeyError:
            raise ValueError(f"Timestamp {timestamp} not found in time series.")

    def get_timestamp_at_value(self, value, variable_name=None):
        """
        Get the timestamp(s) where the value is equal to the specified value for a given variable.

        :param value: The value to search for.
        :param variable_name: The name of the variable to check (optional).
        :return: A list of timestamps where the value is found.
        """
        if variable_name:
            matching_times = self.data[variable_name].where(self.data[variable_name] == value, drop=True).coords['time']
        else:
            matching_times = self.data.where(self.data == value, drop=True).coords['time']
        if matching_times.size == 0:
            raise ValueError(f"No matching timestamp found for value {value}.")
        return matching_times.values

    def last_value(self,variable_name=None):
        """
           Get the last value and its timestamp from the time series for a specified variable (optional).
           """
        last_timestamp = self.data.time[-1].values
        if variable_name:
            last_value = self.data[variable_name].sel(time=last_timestamp).values
        else:
            last_value = self.data.sel(time=last_timestamp).values
        return last_timestamp, last_value
    def first_value(self, variable_name=None):
        """
        Get the first recorded value for the specified variable.

        :param variable_name: The name of the variable to retrieve (optional).
        :return: The first value recorded for the variable.
        """
        first_timestamp = self.data.time[0].values
        if variable_name:
            first_value = self.data[variable_name].sel(time=first_timestamp).values
        else:
            first_value = self.data.sel(time=first_timestamp).values
        return first_timestamp, first_value
    def limit(self, num_points, order='last'):
        """
        Limit the time series to a specific number of points.

        Parameters:
        - num_points (int): Number of points to limit the data to.
        - order (str): Whether to get the 'first' or 'last' points. Default is 'last'.

        Returns:
        - A subset of the time series data limited by the number of points.
        """
        if order == 'first':
            limited_data = self.data.isel(time=slice(0, num_points))
        elif order == 'last':
            limited_data = self.data.isel(time=slice(-num_points, None))
        else:
            raise ValueError("Order must be either 'first' or 'last'.")
        return limited_data

    def last_timestamp(self):
        """
        Returns the last (most recent) timestamp in the time series.
        """
        return self.data.coords['time'].values[-1]

    def first_timestamp(self):
        """
        Returns the first (earliest) timestamp in the time series.
        """
        return self.data.coords['time'].values[0]

    def has_timestamp(self, timestamp):
        """
        Check if a specific timestamp exists in the time series.

        :param timestamp: The timestamp to check
        :return: True if the timestamp exists, False otherwise
        """
        timestamp = pd.Timestamp(timestamp).to_datetime64()  # Ensure consistent format
        return timestamp in self.data.coords['time'].values

    def update_value_at_timestamp(self, timestamp, new_value, variable_name=None):
        """
        Update the value at a specific timestamp in the time series for a specified variable.

        :param timestamp: The timestamp to update
        :param new_value: The new value to set at the given timestamp
        :param variable_name: The name of the variable to update (optional for multivariate series)
        :raises ValueError: If the timestamp or variable is not found in the time series
        """
        timestamp = pd.Timestamp(timestamp).to_datetime64()  # Ensure consistent format

        # Check if the timestamp exists in the time series
        if timestamp not in self.data.coords['time'].values:
            raise ValueError(f"Timestamp {timestamp} not found in time series.")

        # If variable_name is provided, check if it's in the data's variables
        if variable_name:
            if variable_name not in self.data.coords['variable'].values:
                raise ValueError(f"Variable '{variable_name}' not found in time series.")
            # Update the value for the specified variable at the given timestamp
            self.data.loc[{'time': timestamp, 'variable': variable_name}] = new_value
        else:
            # For univariate series, update without specifying variable
            if len(self.data.coords['variable'].values) > 1:
                raise ValueError("Time series is multivariate; specify a variable_name.")
            self.data.loc[{'time': timestamp}] = new_value

    def display_time_series(self, limit=10, order='first'):
        """
               Display the time series data with optional limits on the number of data points.

               Parameters:
               - limit (int, optional): The maximum number of data points to display.
               - order (str, optional): Whether to retrieve 'first' or 'last' N data points.
               """
        # Display the time series metadata
        if self.metadata.owner_id == -1:
            print(f"Time Series {self.tsid}")
        else:
            print(f"Time Series {self.tsid}: {self.metadata.owner_id}")

        variables = [str(var) for var in self.data.coords['variable'].values]
        print(f"Variables: {', '.join(variables)}")

        # Convert to DataFrame and limit data if needed
        ts_df = self.data.to_dataframe('value').reset_index()
        if limit is not None:
            if order == 'first':
                ts_df = ts_df.head(limit)
            elif order == 'last':
                ts_df = ts_df.tail(limit)

        # Group by time and generate the output
        grouped = ts_df.groupby('time')

        output = []
        for time, group in grouped:
            values = [f"{row['value']}" for idx, row in group.iterrows()]
            row_str = ", ".join(values)
            output.append(f"{time}, {row_str}")

        # Print the output once after processing all groups
        for line in output:
            print(line)
        return output

    # --- Time Series Similarity ---
    def euclidean_distance(self, other_timeseries):
        """
        Compute the Euclidean distance between this time series and another.
        Both time series must have the same length.
        """
        self_values = self.data.values
        other_values = other_timeseries.data.values
        if self_values.shape != other_values.shape:
            raise ValueError("Both time series must have the same length for Euclidean distance.")
        return np.linalg.norm(self_values - other_values)

    def correlation_coefficient(self, other_timeseries):
        self_values = self.data.values
        other_values = other_timeseries.data.values
        if self_values.shape != other_values.shape:
            raise ValueError("Both time series must have the same length for calculating correlation coefficient.")
        correlation = np.corrcoef(self_values, other_values)[0, 1]
        return correlation

    def cosine_similarity(self, other_timeseries):
        self_values = self.data.values.flatten()
        other_values = other_timeseries.data.values.flatten()
        if np.any(np.linalg.norm(self_values) == 0) or np.any(np.linalg.norm(other_values) == 0):
            raise ValueError("One of the time series is a zero vector which makes cosine similarity undefined.")
        cosine_value = np.dot(self_values, other_values) / (np.linalg.norm(self_values) * np.linalg.norm(other_values))
        return cosine_value

    def manhattan_distance(self, other_timeseries):
        self_values = self.data.values
        other_values = other_timeseries.data.values
        if self_values.shape != other_values.shape:
            raise ValueError("Both time series must have the same length for Manhattan distance.")
        return np.sum(np.abs(self_values - other_values))

    def dynamic_time_warping(self, other_timeseries):
        """
        Compute the Dynamic Time Warping (DTW) distance between two time series.
        """
        self_df = self.to_dataframe()
        other_df = other_timeseries.to_dataframe()
        distance, _ = fastdtw(self_df['value'].values, other_df['value'].values, dist=euclidean)
        return distance

    # --- Time Series Classification ---
    def classify(self, train_data, train_labels, method='knn', **kwargs):
        """
        Train a classifier on time series data and classify this time series.
        Parameters:
        - train_data: List of time series data (each as a pandas DataFrame).
        - train_labels: List of labels corresponding to each training time series.
        - method: Classifier method ('knn', 'random_forest', 'svm').
        - kwargs: Additional arguments to pass to the classifier.
        """
        # Convert each time series to a flattened numpy array for classification
        train_data_flat = [ts['value'].values.flatten() for ts in train_data]
        test_data = self.to_dataframe()['value'].values.flatten().reshape(1, -1)

        # Select the classifier
        if method == 'knn':
            classifier = KNeighborsClassifier(**kwargs)
        elif method == 'random_forest':
            classifier = RandomForestClassifier(**kwargs)
        elif method == 'svm':
            classifier = SVC(**kwargs)
        else:
            raise ValueError(f"Unsupported classification method: {method}")

        # Train the classifier
        classifier.fit(train_data_flat, train_labels)

        # Predict the label for the current time series
        return classifier.predict(test_data)

    # --- Helper Functions ---
    def to_dataframe(self):
        """
        Convert the time-series to a pandas DataFrame.
        """
        return self.data.to_dataframe('value').reset_index()

    def autocorrelation(self, lag=1):
        """
        Compute the autocorrelation for a given lag.
        """
        df = self.to_dataframe().set_index('time')
        return df['value'].autocorr(lag=lag)

    def check_stationarity(self):
        """
        Perform the Augmented Dickey-Fuller test to check for stationarity.
        Returns the p-value of the test.
        """
        df = self.to_dataframe()
        result = adfuller(df['value'].values)
        return result[1]  # p-value

    def exponential_smoothing(self, trend=None, seasonal=None, seasonal_periods=None):
        """
        Apply Holt-Winters Exponential Smoothing.
        """
        df = self.to_dataframe().set_index('time')
        model = ExponentialSmoothing(df['value'], trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods)
        fitted_model = model.fit()
        return fitted_model.fittedvalues


def generate_time_series(length, noise_factor=0.1):
    """
    Generate a sine wave time series with added noise.
    :param length: Length of the time series.
    :param noise_factor: Amplitude of the noise to add.
    :return: Generated time series.
    """
    t = np.linspace(0, 4 * np.pi, length)
    series = np.sin(t) + noise_factor * np.random.randn(length)
    return series

def plot_time_series(ts1, ts2, title):
    """
    Plot two time series on the same graph.
    :param ts1: First time series.
    :param ts2: Second time series.
    :param title: Title of the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(ts1, label='Time Series 1')
    plt.plot(ts2, label='Time Series 2')
    plt.title(title)
    plt.legend()
    plt.show()

def compute_dtw(ts1, ts2):
    """
    Compute the DTW distance between two time series.
    :param ts1: First time series.
    :param ts2: Second time series.
    :return: DTW distance and the alignment path.
    """
    distance, path = fastdtw(ts1, ts2, dist=euclidean)
    return distance, path

def main():
    # Generate similar time series
    ts1 = generate_time_series(100)
    ts2 = generate_time_series(100, noise_factor=0.1)

    # Plot similar time series
    plot_time_series(ts1, ts2, "Similar Time Series")

    # Compute DTW distance for similar time series
    distance, path = compute_dtw(ts1, ts2)
    print(f"DTW distance for similar time series: {distance}")

    # Generate different time series
    ts3 = generate_time_series(100)
    ts4 = generate_time_series(100, noise_factor=0.5) + 2  # Adding an offset to make them different

    # Plot different time series
    plot_time_series(ts3, ts4, "Different Time Series")

    # Compute DTW distance for different time series
    distance, path = compute_dtw(ts3, ts4)
    print(f"DTW distance for different time series: {distance}")

if __name__ == "__main__":
    main()









































