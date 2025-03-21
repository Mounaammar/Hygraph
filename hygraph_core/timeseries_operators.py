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
    def __init__(self, owner_id, element_type='',attributes=None):
        self.owner_id = owner_id
        self.element_type = element_type
        self.attributes = attributes or {}  # Initialize attributes as a dictionary

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
        time_index = pd.to_datetime(timestamps, infer_datetime_format=True)
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
        return self.tsid

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

    def aggregate_time_series_cumulative(self, ts2, result_variable_name):
        """
        Aggregate two time series to maintain a cumulative count across time, regardless of variable names.

        :param ts2: Second time series (TimeSeries instance).
        :param result_variable_name: Name for the resulting variable in the aggregated time series.
        :return: Aggregated time series with cumulative values as a new TimeSeries instance.
        """
        # Collect all unique timestamps from both time series
        timestamps = sorted(set(self.data.time.values).union(set(ts2.data.time.values)))

        # Initialize cumulative values array
        cumulative_values = np.zeros(len(timestamps),dtype=int)
        cumulative_value = 0

        # Calculate cumulative value at each timestamp
        for i,timestamp in enumerate(timestamps):
            # Fetch values from each series, defaulting to 0 if timestamp is missing
            value1 = self.get_value_at_timestamp(timestamp) if self.has_timestamp(timestamp) else 0
            value2 = ts2.get_value_at_timestamp(timestamp) if ts2.has_timestamp(timestamp) else 0

            # Increment cumulative sum and store for current timestamp
            #cumulative_value += value1 + value2
            cumulative_values[i]=value1 + value2

        # Create a DataArray for cumulative results with aligned timestamps
        data_array = xr.DataArray(
            np.array(cumulative_values).reshape(-1, 1),
            coords={'time': pd.to_datetime(timestamps), 'variable': [result_variable_name]},
            dims=['time', 'variable']
        )

        # Return new TimeSeries with cumulative data
        metadata = TimeSeriesMetadata(owner_id=None)
        tsid = f"{self.tsid}_cumulative_{ts2.tsid}"
        return TimeSeries(tsid, data_array.time, [result_variable_name], data_array.values, metadata)

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

    '''def get_value_at_timestamp(self, timestamp, variable_name=None):
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
            raise ValueError(f"Timestamp {timestamp} not found in time series.")'''

    def get_value_at_timestamp(self, timestamp, variable_name=None):
        """
        Retrieve the value at a specific timestamp for the given variable.

        :param timestamp: Timestamp for which the value is needed.
        :param variable_name: The name of the variable to retrieve (optional).
        :return: The value at the given timestamp.
        """
        # Ensure timestamp is a numpy datetime64 for compatibility with times array
        timestamp = np.datetime64(pd.to_datetime(timestamp))
        times = self.data.coords['time'].values

        # Find the index of the timestamp or the most recent timestamp before it
        if timestamp < times[0]:
            # The timestamp is before the earliest timestamp in the data
            return None
        idx = np.searchsorted(times, timestamp, side='right') - 1

        if variable_name:
            # Retrieve the value for the specific variable
            try:
                return self.data.isel(time=idx).sel(variable=variable_name).values.item()
            except KeyError:
                raise ValueError(f"Variable '{variable_name}' not found in time series.")
        else:
            # Retrieve the value for the first variable if no variable_name is specified
            return self.data.isel(time=idx, variable=0).values.item()

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
        # Ensure the variables and time indices match
        if not np.array_equal(self.data.coords['variable'], other_timeseries.data.coords['variable']):
            raise ValueError("TimeSeries objects have different variables.")
        if not np.array_equal(self.data.coords['time'], other_timeseries.data.coords['time']):
            raise ValueError("TimeSeries objects have different time indices.")

        # Extract data as numpy arrays and flatten
        data1 = self.data.values.flatten()
        data2 = other_timeseries.data.values.flatten()

        # Min-Max normalization across both datasets
        total_min = min(data1.min(), data2.min())
        total_max = max(data1.max(), data2.max())
        range_total = total_max - total_min

        if range_total == 0:
            # Avoid division by zero if all values are the same
            data1_norm = data1 * 0
            data2_norm = data2 * 0
        else:
            data1_norm = (data1 - total_min) / range_total
            data2_norm = (data2 - total_min) / range_total

        # Compute Euclidean distance
        distance = np.linalg.norm(data1_norm - data2_norm)

        # Compute maximum possible distance for normalization
        max_distance = np.sqrt(len(data1_norm))
        if max_distance == 0:
            # Handle the case where there are no elements
            normalized_distance = 0.0
        else:
            normalized_distance = distance / max_distance

        return normalized_distance

    def correlation_coefficient(self, other_timeseries):
        self_values = self.data.values
        other_values = other_timeseries.data.values

        if self_values.shape != other_values.shape:
            # Truncate to the minimum length
            min_length = min(self_values.shape[0], other_values.shape[0])
            self_values = self_values[:min_length]
            other_values = other_values[:min_length]

        # Flatten the arrays
        self_flat = self_values.flatten()
        other_flat = other_values.flatten()

        # Compute correlation coefficient
        correlation_matrix = np.corrcoef(self_flat, other_flat)
        correlation = correlation_matrix[0, 1]
        return correlation

    def cosine_similarity(self, other_timeseries):
        self_values = self.data.values.flatten()
        other_values = other_timeseries.data.values.flatten()

        norm_self = np.linalg.norm(self_values)
        norm_other = np.linalg.norm(other_values)

        if norm_self == 0 or norm_other == 0:
            raise ValueError("One of the time series is a zero vector, which makes cosine similarity undefined.")

        cosine_value = np.dot(self_values, other_values) / (norm_self * norm_other)
        return cosine_value

    def dynamic_time_warping(ts1, ts2, variable_name):
        """
        Compute normalized DTW distance between two TimeSeries instances.
        """
        # Align the data based on timestamps
        df1 = ts1.data.sel(variable=variable_name).to_pandas()
        df2 = ts2.data.sel(variable=variable_name).to_pandas()

        # Since DTW does not require equal lengths, we can proceed directly
        # However, we need the data as numpy arrays
        x = df1.values
        y = df2.values

        # Compute raw DTW distance
        distance, path = fastdtw(x, y, dist=lambda a, b: np.abs(a - b))

        # Normalize the distance
        # Maximum possible distance is the sum over the alignment path of the maximum possible difference
        max_val = max(df1.max(), df2.max())
        min_val = min(df1.min(), df2.min())
        max_possible_distance = len(path) * (max_val - min_val)
        if max_possible_distance == 0:
            normalized_distance = 0.0
        else:
            normalized_distance = distance / max_possible_distance

        return normalized_distance

    def dtw_independent_multivariate(self, other_timeseries, window: int = None) -> float:
        """
        Compute the normalized DTW distance by summing DTW distances over each variable.

        :param other_timeseries: Another TimeSeries instance
        :param window: Maximum warping window size (int). If None, no constraint.
        :return: Normalized total DTW distance across all variables (float)
        """
        variables = self.data.coords['variable'].values
        total_distance = 0.0

        for var in variables:
            series1 = self.data.sel(variable=var).values.flatten()
            series2 = other_timeseries.data.sel(variable=var).values.flatten()

            # Handle different lengths by trimming or interpolation if needed
            min_length = min(len(series1), len(series2))
            series1 = series1[:min_length]
            series2 = series2[:min_length]

            # Normalize the series
            min_val = min(series1.min(), series2.min())
            max_val = max(series1.max(), series2.max())
            series1_norm = (series1 - min_val) / (max_val - min_val)
            series2_norm = (series2 - min_val) / (max_val - min_val)

            # Compute DTW distance using absolute difference
            distance, _ = fastdtw(series1_norm, series2_norm, dist=lambda x, y: abs(x - y))
            total_distance += distance

        # Normalize the total distance
        normalized_distance = total_distance / (len(variables) * min_length)
        return normalized_distance

    def dtw_dependent_multivariate(self, other_timeseries, window: int = None) -> float:
        """
        Compute the normalized DTW distance by considering variables together at each time point.

        :param other_timeseries: Another TimeSeries instance
        :param window: Maximum warping window size (int). If None, no constraint.
        :return: Normalized DTW distance (float)
        """
        # Extract data as sequences of vectors
        series1 = self.data.values
        series2 = other_timeseries.data.values

        # Handle different lengths by trimming or interpolation if needed
        min_length = min(series1.shape[0], series2.shape[0])
        series1 = series1[:min_length]
        series2 = series2[:min_length]

        # Normalize the data
        min_val = min(series1.min(), series2.min())
        max_val = max(series1.max(), series2.max())
        series1_norm = (series1 - min_val) / (max_val - min_val)
        series2_norm = (series2 - min_val) / (max_val - min_val)

        # Compute DTW distance using Euclidean distance between vectors
        distance, _ = fastdtw(series1_norm, series2_norm, dist=euclidean)
        # Normalize by the sequence length
        normalized_distance = distance / min_length
        return normalized_distance

    def extract_features(self):
        """
        Extract statistical features from the time series data.
        :return: Feature vector (numpy array)
        """
        data = self.data.values.flatten()
        features = []

        # Compute statistical features
        features.append(np.mean(data))  # Mean
        features.append(np.std(data))  # Standard deviation
        features.append(np.min(data))  # Minimum value
        features.append(np.max(data))  # Maximum value
        features.append(np.median(data))  # Median
        features.append(np.percentile(data, 25))  # 25th percentile
        features.append(np.percentile(data, 75))  # 75th percentile
        features.append(np.var(data))  # Variance
        # Add more features as needed

        # Convert features list to numpy array
        features_vector = np.array(features)
        return features_vector

    def feature_similarity(self, other_timeseries, metric='cosine'):
        """
        Compute feature similarity between this time series and another.
        :param other_timeseries: Another TimeSeries object
        :param metric: Similarity metric ('cosine', 'euclidean', etc.)
        :return: Similarity value
        """
        features1 = self.extract_features()
        features2 = other_timeseries.extract_features()

        if metric == 'cosine':
            similarity = self.cosine_similarity_features(features1, features2)
        elif metric == 'euclidean':
            distance = np.linalg.norm(features1 - features2)
            similarity = 1 / (1 + distance)  # Convert distance to similarity
        else:
            raise ValueError(f"Unsupported feature similarity metric: {metric}")
        return similarity

    def cosine_similarity_features(self, vec1, vec2):
        """
        Compute cosine similarity between two feature vectors.
        :param vec1: Feature vector from this time series
        :param vec2: Feature vector from another time series
        :return: Cosine similarity value
        """
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        similarity = np.dot(vec1, vec2) / (norm1 * norm2)
        return similarity

    def shape_similarity(self, other_timeseries, variable_name, metric='euclidean',):
        """
        Compute shape similarity using existing methods.
        :param other_timeseries: Another TimeSeries object
        :param metric: Similarity metric
        :return: Similarity value
        """
        if metric == 'euclidean':
            distance = self.euclidean_distance(other_timeseries)
            similarity = 1 / (1 + distance)
        elif metric == 'dtw':
            distance = self.dynamic_time_warping(other_timeseries,variable_name)
            similarity = 1 / (1 + distance)
        elif metric == 'correlation':
            similarity = self.correlation_coefficient(other_timeseries)
        elif metric == 'cosine':
            similarity = self.cosine_similarity(other_timeseries)
        else:
            raise ValueError(f"Unsupported shape similarity metric: {metric}")
        return similarity
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
    #Utility functions
    def align_time_series(self, other):
        """
        Align two time series on their timestamps and variables.

        :param other: Another TimeSeries object
        :return: Tuple of aligned TimeSeries objects
        """
        # Find common timestamps
        common_times = self.data.coords['time'].values
        common_vars = self.data.coords['variable'].values

        # Reindex both time series
        ts1_aligned = self.data.sel(time=common_times, variable=common_vars)
        ts2_aligned = other.data.sel(time=common_times, variable=common_vars)

        return ts1_aligned, ts2_aligned










































