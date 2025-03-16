from datetime import datetime

import numpy as np
import pandas as pd
from hygraph_core.timeseries_operators import TimeSeries, TimeSeriesMetadata


class GraphElement:
    """
    Superclass for all graph elements (node, edge, subgraph) with shared property management.
    """

    def __init__(self, oid, label, element_type, static_properties=None, temporal_properties=None):
        self.oid = oid
        self.label = label
        self.element_type = element_type
        self.static_properties = static_properties if static_properties is not None else {}
        self.temporal_properties = temporal_properties if temporal_properties is not None else {}

    def __repr__(self):

        return (f"{self.element_type}: (oid={self.oid}, label='{self.label}', "
                f"properties={(self.get_all_properties(False, 10))}, "
                )

    def get_property_type(self, property_name):
        """
        Determine the type of the property (static or dynamic).
        """
        if property_name in self.static_properties:
            return 'static'
        elif property_name in self.temporal_properties:
            return 'dynamic'
        else:
            return None

    def add_static_property(self, name, value,hygraph):
        self.static_properties[name] = StaticProperty(name, value)
        # Now call the shared function to update the graph
        self.update_graph_element(hygraph)

    def add_temporal_property(self, name, timeseries, hygraph):
        """
        Add a new temporal property to the graph element.

        :param name: Name of the temporal property
        :param timeseries: A timeseries that is already inserted in HyGraph

        """
        if name in self.temporal_properties:
            raise ValueError(f"Temporal property '{name}' already exists.")

        timeseries.metadata.update_metadata(self.oid, self.element_type)
        # Create a TemporalProperty and add it to the element's temporal_properties
        temporal_property = TemporalProperty(name, timeseries.tsid, hygraph)
        self.temporal_properties[name] = temporal_property
        # Now call the shared function to update the graph
        self.update_graph_element(hygraph)
        #print(f"Temporal property '{name}' added with time series ID {timeseries.tsid}")

    def get_static_property(self, name):
        if name in self.static_properties:
            return self.static_properties[name].get_value()
        else:
            raise ValueError(f"Static property {name} not found.")

    def get_temporal_property(self, name, limit=None, order='first'):
        """
        Retrieve the specified temporal property (dynamic property) with an option to limit the number of data points.

        Parameters:
        - name (str): The name of the property to retrieve.
        - limit (int, optional): The maximum number of data points to retrieve.
        - order (str, optional): Whether to retrieve the 'first' or 'last' N data points. Defaults to 'first'.
        """
        if name in self.temporal_properties:
            timeseries = self.temporal_properties[name].get_time_series()

            # Call the display method to show time series data
            ts=timeseries.display_time_series(limit=limit, order=order)

            return timeseries
        else:
            raise ValueError(f"Dynamic property {name} not found.")

    '''   def get_all_properties(self, display_ts=False, limit=None, order='first'):
        """
            Retrieve and display both static and temporal properties of the graph element.

            Parameters:
            - limit (int, optional): The maximum number of data points to retrieve for temporal properties.
            - order (str, optional): Whether to retrieve the 'first' or 'last' N data points for temporal properties. Defaults to 'first'.
            """
        print("Static Properties:")
        for name, prop in self.static_properties.items():
            print(f"{name}: {prop.get_value()} (Last updated: {prop.timestamp})")

        print("\nTemporal Properties:")
        for name, ts in self.temporal_properties.items():
            if display_ts:
                ts.display_time_series(limit, order)
            else:
                print(f"{name}: TimeSeries ID: {ts.tsid}")'''

    def get_all_properties(self, display_ts=False, limit=None, order='first'):
        """
        Retrieve and display both static and temporal properties of the graph element.

        Parameters:
        - limit (int, optional): The maximum number of data points to retrieve for temporal properties.
        - order (str, optional): Whether to retrieve the 'first' or 'last' N data points for temporal properties. Defaults to 'first'.
        """
        static_props = {}
        temporal_props = {}

        # Static properties
        for name, prop in self.static_properties.items():
            static_props[name] = str(prop)

        # Temporal properties
        for name, ts in self.temporal_properties.items():
            if display_ts:
                ts.get_time_series().display_time_series(limit, order)
            else:
                temporal_props[name] = f"TimeSeries ID: {ts}"

        return {'static': static_props, 'temporal': temporal_props}

    def get_property(self, property_name, limit=None, order='last'):
        """
        Retrieve either static or temporal property.
        """
        if property_name in self.static_properties:
            return self.get_static_property(property_name)
        elif property_name in self.temporal_properties:
            return self.get_temporal_property(property_name, limit, order)
        else:
            raise ValueError(f"Property {property_name} not found.")

    def _get_static_properties(self):
        """
        Retrieve all static properties from the element.
        """
        return {key: value for key, value in self.static_properties.items()}

    def _get_temporal_properties(self):
        """
        Retrieve all temporal (dynamic) properties (time-series) from the element.
        """
        return {key: value for key, value in self.temporal_properties.items()}

    def getId(self):
        """
        Get the ID of the graph element.
        """
        return self.oid

    def getLabel(self):
        """
        Get the label of the graph element.
        """
        return self.label

    # utility function
    def update_graph_element(self, hygraph):
        """
        Update the graph element in the hygraph (either PGNode or PGEdge).
        """
        if self.oid in hygraph.graph.nodes:
            # Update a PGNode in the graph
            hygraph.graph.add_node(self.oid,
                                   label=self.label,
                                   start_time=getattr(self, 'start_time', None),
                                   end_time=getattr(self, 'end_time', None),
                                   properties={**self.static_properties, **self.temporal_properties},
                                   membership=getattr(self, 'membership', None),
                                   type="PGNode")
            #print(f"Updated PGNode {self.oid} in the graph.")
        else:
            # Update a PGEdge in the graph
            for u, v, key, edge_data in hygraph.graph.edges(data=True, keys=True):
                if edge_data.get('oid') == self.oid:
                    hygraph.graph.add_edge(u, v,
                                           key=key,
                                           oid=self.oid,
                                           label=self.label,
                                           start_time=getattr(self, 'start_time', None),
                                           end_time=getattr(self, 'end_time', None),
                                           properties={**self.static_properties, **self.temporal_properties},
                                           membership=getattr(self, 'membership', None),
                                           type="PGEdge")
                    #print(f"Updated PGEdge {self.oid} from {u} to {v} in the graph.")
                    break


class Node(GraphElement):
    def __init__(self, oid, label, element_type, membership=None, static_propeprties=None,
                 temporal_properties=None, node_id=None):
        super().__init__(oid, label, element_type, static_propeprties,
                         temporal_properties)  # Initialize the Subject
        self.node_id = node_id  # External ID from CSV file
        self.membership = membership

    def __repr__(self):
        base_str = super().__repr__()
        return f"Node({base_str}, label={self.label}, membership={self.membership})"

    def get_type(self):
        """
        Returns the type of node. Should be overridden by subclasses.
        """
        return "Node"

    def get_membership(self):
        """
        Get the membership time series of the element.
        """
        if self.membership:
            tsid = self.membership  # Membership is a time series ID
            if tsid in self.hygraph.time_series:
                return self.hygraph.time_series[tsid]
            else:
                return None
        return None

    def get_neighbors(self, hygraph,direction='both',):
        neighbors = []
        if direction in ('both', 'out'):
            neighbors.extend((target, edge) for target, edge in hygraph.graph.out_edges(self, data=True))
        if direction in ('both', 'in'):
            neighbors.extend((source, edge) for source, edge in hygraph.graph.in_edges(self, data=True))
        return neighbors


class PGNode(Node):
    def __init__(self, oid, label, start_time,end_time, element_type='PGNode', node_id=None, ):
        super().__init__(oid, label, element_type, node_id)
        self.start_time = start_time
        self.end_time = end_time

    def __repr__(self):
        base_repr = super().__repr__()
        start_time_str = BaseTimeFormatter.format_time(self.start_time)
        end_time_str = BaseTimeFormatter.format_time(self.end_time)

        return f"{base_repr}, start_time={start_time_str}, end_time={end_time_str}"

    def get_type(self):
        return "PGNode"


class TSNode(Node):
    def __init__(self, oid, label, time_series, element_type='TSNode', node_id=None):
        super().__init__(oid, label, element_type, node_id)
        self.series = time_series  # TimeSeries object

    def __repr__(self):
        base_str = super().__repr__()
        return f"{base_str}, series={self.series.tsid}"

    def get_type(self):
        return "TSNode"

    # Override property methods to raise an error if attempted
    def add_static_property(self, *args, **kwargs):
        raise AttributeError("TSNode does not support static properties.")

    def add_temporal_property(self, *args, **kwargs):
        raise AttributeError("TSNode does not support temporal properties.")


class Edge(GraphElement):

    def __init__(self, oid, source, target, label, element_type, membership=None, static_properties=None,
                 temporal_properties=None, edge_id=None):
        super().__init__(oid, label, element_type, static_properties, temporal_properties)
        self.membership = membership
        self.source = source
        self.target = target
        self.edge_id = edge_id  # External ID from CSV file

    def __repr__(self):
        base_repr = super().__repr__()
        return f"Edge({base_repr}, source={self.source}, target={self.target}, membership={self.membership} )"


class PGEdge(Edge):
    def __init__(self, oid, source, target,label, start_time, end_time,element_type='PGEdge', edge_id=None):
        super().__init__(oid,source, target, label, element_type, edge_id)
        self.start_time = start_time
        self.end_time = end_time

    def __repr__(self):
        base_str = super().__repr__()
        start_time_str = BaseTimeFormatter.format_time(self.start_time)
        end_time_str = BaseTimeFormatter.format_time(self.end_time)

        return f"{base_str}, start_time={start_time_str}, end_time={end_time_str}}}"


class TSEdge(Edge):
    def __init__(self, oid, source, target, label, time_series, element_type='TSEdge', edge_id=None):
        super().__init__(oid, source, target, label, element_type, edge_id)
        self.series = time_series  # TimeSeries object


    def __repr__(self):
        base_str = super().__repr__()
        return f"{base_str}, series={self.series.tsid}"

    def get_type(self):
        return "TSEdge"

    # Override property methods to raise an error if attempted
    def add_static_property(self, *args, **kwargs):
        raise AttributeError("TSEdge does not support static properties.")

    def add_temporal_property(self, *args, **kwargs):
        raise AttributeError("TSEdge does not support temporal properties.")


class Subgraph(GraphElement):
    def __init__(self, subgraph_id, label, start_time, end_time=None, static_properties=None,
                 temporal_properties=None, filter_func=None):
        # Initialize the base class with oid, label, and hygraph
        super().__init__(subgraph_id, label, static_properties, temporal_properties)
        self.start_time = start_time
        self.end_time = end_time
        self.filter_func = filter_func  # Function to filter nodes or edges that belong to this subgraph

    def __repr__(self):
        # Call the base class's __repr__ to include static and temporal properties
        base_repr = super().__repr__()
        start_time_str = BaseTimeFormatter.format_time(self.start_time)
        end_time_str = BaseTimeFormatter.format_time(self.end_time)
        return f"{base_repr}, start_time={start_time_str}, end_time={end_time_str}"

    def get_type(self):
        # Identify this as a subgraph
        return "Subgraph"

    def apply_filter(self, element):
        """
        Apply the subgraph's filter function to determine if an element belongs to the subgraph.
        """
        if self.filter_func:
            return self.filter_func(element)
        return False  # Default behavior if no filter function is provided



class StaticProperty:
    def __init__(self, name, value):
        self.last_updated = None
        self.name = name
        self.value = value

    def get_value(self):
        return self.value

    def set_value(self, new_value):
        self.value = new_value
        self.last_updated = datetime.now()

    def __repr__(self):
        return f" (value={self.value}, last_updated={self.last_updated})"

    def __str__(self):
        return f"({self.value} (Last updated: {self.last_updated})"


class TemporalProperty:
    def __init__(self, name, time_series_id, hygraph):
        self.name = name
        self.hygraph = hygraph
        # assert isinstance(time_series, TimeSeries), "time_series must be of type TimeSeries"
        self.time_series_id = time_series_id  # Instance of TimeSeries class

    def get_time_series(self) -> TimeSeries:
        return self.hygraph.time_series.get(self.time_series_id)

    def set_value(self, timestamp, value):
        """
        Add a new entry to the temporal property's time series.

        :param hygraph: The HyGraph instance containing the time series.
        :param timestamp: The timestamp for the new entry.
        :param value: The value to insert.
        """
        time_series = self.get_time_series()
        if time_series is None:
            raise ValueError(f"Time series with ID {self.time_series_id} does not exist.")

        timestamp = pd.to_datetime(timestamp)

        # Check if timestamp already exists
        existing_timestamps = time_series.data.coords['time'].values
        if timestamp in existing_timestamps:
            raise ValueError(f"Timestamp {timestamp} already exists in the time series.")

        # Use the append_data method, which handles timestamp checks
        try:
            time_series.append_data(timestamp, value)
            #print(f"Value {value} added at timestamp {timestamp} to time series {self.time_series_id}.")
        except ValueError as e:
            print(f"Error adding value: {e}")

    def get_value_at(self, timestamp,variable_name):
        return self.get_time_series().get_value_at_timestamp(timestamp,variable_name)

    def apply_aggregation(self, function_name, start_time=None, end_time=None):
        """
            Apply an aggregation function (like 'sum', 'mean', etc.) to the time series data,
            optionally filtered by a time range (start_time to end_time).

            :param function_name: Aggregation function ('sum', 'mean', 'min', 'max', 'count', etc.)
            :param start_time: Start of the time range for filtering (optional).
            :param end_time: End of the time range for filtering (optional).
            :return: Aggregated value based on the specified function.
            """
        filtered_data = self.get_time_series().data
        if start_time and end_time:
            # Filter data within the given time range
            filtered_data = self.get_time_series().data.sel(time=slice(start_time, end_time))
        elif start_time and not end_time:
            # Filter data starting from start_time to the last available timestamp
            filtered_data = self.get_time_series().data.sel(time=slice(start_time, None))
        elif not start_time and end_time:
            # Filter data from the first available timestamp to end_time
            filtered_data = self.get_time_series().data.sel(time=slice(None, end_time))

        return filtered_data.apply_aggregation(function_name)

    def get_timestamp(self, value,variable_name):
        return self.get_time_series().get_timestamp_at_value(value,variable_name)

    def get_last_value(self,variable_name):
        return self.get_time_series().last_value(variable_name)

    def get_first_value(self,variable_name):
        return self.get_time_series().first_value(variable_name)

    def __repr__(self):
        time_series = self.get_time_series()
        last_value = time_series.last_value() if time_series else "No time series"
        first_value = time_series.first_value() if time_series else "No time series"
        return (f" (TimeSeries ID={self.time_series_id}, "
                f"First value: ({first_value[0]}, {first_value[1]}), "
                f"Last value: ({last_value[0]}, {last_value[1]})")

    def __str__(self):
        return self.__repr__()


#UTILITY FUNCTIONS
class BaseTimeFormatter:
    @staticmethod
    def format_time(time):
        if isinstance(time, np.datetime64):
            time = pd.to_datetime(time).to_pydatetime()
        return time.isoformat() if time else 'None'