from itertools import product

import networkx as nx
import numpy as np
import pandas as pd
import xarray as xr
from fastdtw import fastdtw

from scipy.spatial.distance import euclidean
from datetime import datetime, timedelta
from igraph import Graph as IGraph
from collections import deque
from collections import defaultdict
from hygraph_core.graph_operators import Edge, TSNode, PGNode, PGEdge, TSEdge, Subgraph, TemporalProperty, \
    StaticProperty
from hygraph_core.timeseries_operators import TimeSeries, TimeSeriesMetadata
from hygraph_core.idGenerator import IDGenerator

FAR_FUTURE_DATE = datetime(2100, 12, 31, 23, 59, 59)
from hygraph_core.constraints import parse_datetime


class HyGraph:
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.time_series = {}
        self.subgraphs = {}
        self.id_generator = IDGenerator()
        self.updated = False  # Flag to track updates
        self.query =None

    #create

    def initialize_degree_timeseries(self, oid, start_time):
        for degree_type in ['in_degree', 'out_degree']:
            tsid = self.id_generator.generate_timeseries_id()
            metadata = TimeSeriesMetadata(owner_id=oid, element_type='node')
            degree_ts = TimeSeries(tsid=tsid, timestamps=[start_time], variables=[degree_type], data=[[0]],
                                   metadata=metadata)

            # Add this time series to the time_series dictionary and store it in metadata
            self.time_series[tsid] = degree_ts
            self.graph.nodes[oid]['metadata'][degree_type] = degree_ts  # Attach directly to metadata
            #print(f"Initialized {degree_type} as TimeSeries for node {oid} with initial value 0 at {start_time}")


    def add_pgedge(self, oid, source, target, label, start_time,end_time=None, properties=None, membership=None):
        """
        Adds a PGEdge to the Hygraph, supporting both static and temporal properties, and membership.

        :param oid: Edge identifier
        :param source: Source node of the edge
        :param target: Target node of the edge
        :param label: Edge label
        :param start_time: Start time for PGEdge
        :param end_time: End time (optional)
        :param properties: Dictionary of static and temporal properties. Temporal properties should be TimeSeries instances.
        :param membership: Timeseries object representing the membership information.
        """
        # Step 1: Check if source and target nodes exist
        if not self.graph.has_node(source):
            raise ValueError(f"Source node {source} does not exist.")
        if not self.graph.has_node(target):
            raise ValueError(f"Target node {target} does not exist.")

        # Step 2: Create the PGEdge instance
        if end_time is None: end_time=FAR_FUTURE_DATE
        pgedge = PGEdge(oid, source,target,label, start_time, end_time,self)

        # Step 3: Add static and temporal properties
        if properties:
            for prop_name, prop_value in properties.items():
                if isinstance(prop_value, TimeSeries):  # It's a temporal property
                    pgedge.add_temporal_property(prop_name, prop_value,self)
                else:
                    pgedge.add_static_property( prop_name, prop_value,self)

        # Step 4: Add membership (if it exists)
        if membership and isinstance(membership, TimeSeries):
            pgedge.membership = membership.tsid  # Store the timeseries ID for membership
            self.time_series[membership.tsid] = membership  # Add membership timeseries to hygraph storage

        # Step 5: Add the edge to the networkx graph
        self.graph.add_edge(source, target,
                            key=oid,
                            oid=oid,
                            label=label,
                            start_time=start_time,
                            end_time=end_time,
                            properties={**pgedge.static_properties, **pgedge.temporal_properties},
                            # Merging static and temporal properties
                            membership=pgedge.membership,
                            data=pgedge,
                            type="PGEdge")

       # print(f"PGEdge {oid} from {source} to {target} with label '{label}' successfully created.")
        # Mark the graph as updated
        self.set_updated()
        # Update node degree time series
       # self._update_node_degree_time_series(source, target, operation='add', timestamp=start_time)

        return pgedge

    def add_pgnode(self, oid, label, start_time, end_time=None, properties=None, membership=None):
        """
        Adds a PGNode to the Hygraph, supporting both static and temporal properties, and membership.

        :param oid: Node identifier
        :param label: Node label
        :param start_time: Start time for PGNode
        :param end_time: End time (optional)
        :param properties: Dictionary of static and temporal properties. Temporal properties should be TimeSeries instances.
        :param membership: Timeseries object representing the membership information.
        """
        # Step 1: Create the PGNode instance
        if end_time is None : end_time=FAR_FUTURE_DATE
        pgnode = PGNode(oid, label, start_time, end_time)

        # Step 2: Add static and temporal properties
        if properties:
            for prop_name, prop_value in properties.items():
                if isinstance(prop_value, TimeSeries):  # It's a temporal property
                     pgnode.add_temporal_property (prop_name, prop_value,self)
                else:
                    pgnode.add_static_property(prop_name, prop_value,self)

        # Step 3: Add membership (if it exists)
        if membership and isinstance(membership, TimeSeries):
            pgnode.membership = membership.tsid  # Store the timeseries ID for membership
            self.time_series[membership.tsid] = membership  # Add membership timeseries to hygraph storage
            # Initialize in_degree and out_degree time series to 0 at start_time
        metadata = {}
        # Step 4: Add the node to the networkx graph
        self.graph.add_node(oid,
                            label=label,
                            start_time=start_time,
                            end_time=end_time,
                            metadata=metadata,
                            properties={**pgnode.static_properties, **pgnode.temporal_properties},
                            # Merging static and temporal properties
                            membership=pgnode.membership,
                            data=pgnode,
                            type="PGNode")
        # Mark the graph as updated
        self.set_updated()
        # Call function to initialize in_degree and out_degree as TimeSeries within metadata
        #self.initialize_degree_timeseries(oid, start_time)
        #print(f"PGNode {oid} with label '{label}' successfully created and end_time '{pgnode.end_time}'.")


        return pgnode

    def add_tsnode(self, oid, label, time_series):
        """
        Adds a TSNode to the Hygraph. TSNode only has a timeseries and a label, no static properties.

        :param oid: Node identifier
        :param label: Node label
        :param time_series: Timeseries object representing the time series for this node.
        """
        # Step 1: Create the TSNode instance
        tsnode = TSNode(oid, label, time_series)
        metadata = {}
        # Step 2: Add the timeseries to Hygraph storage
        self.time_series[time_series.tsid] = time_series
        start_time = time_series.first_timestamp()
        end_time=time_series.last_timestamp()
        # Step 3: Add the node to the networkx graph
        self.graph.add_node(oid,
                            label=label,
                            series=time_series.tsid,  # Store only the timeseries ID in the node
                            metadata=metadata,
                            data=tsnode,
                            start_time =start_time,
                            end_time=end_time,
                            type="TSNode")
        # Call function to initialize in_degree and out_degree as TimeSeries within metadata
        #self.initialize_degree_timeseries(oid, start_time)
       # print(f"TSNode {oid} with label '{label}' successfully created.")
        # Mark the graph as updated
        self.set_updated()


        return tsnode

    def add_tsedge(self, oid, source, target, label, time_series):
        """
        Adds a TSEdge to the Hygraph. TSEdge only has a timeseries and a label, no static properties.

        :param oid: Edge identifier
        :param source: Source node of the edge
        :param target: Target node of the edge
        :param label: Edge label
        :param time_series: Timeseries object representing the time series for this edge.
        """
        # Step 1: Check if source and target nodes exist
        if not self.graph.has_node(source):
            raise ValueError(f"Source node {source} does not exist.")
        if not self.graph.has_node(target):
            raise ValueError(f"Target node {target} does not exist.")

        # Step 2: Create the TSEdge instance
        tsedge = TSEdge (oid,source,target, label, time_series)

        # Step 3: Add the timeseries to Hygraph storage
        self.time_series[time_series.tsid] = time_series

        # Step 4: Add the edge to the networkx graph
        self.graph.add_edge(source, target,
                            key=oid,
                            oid=oid,
                            label=label,
                            series=time_series.tsid,  # Store only the timeseries ID in the edge
                            start_time=time_series.first_timestamp(),
                            end_time=time_series.last_timestamp(),
                            data=tsedge,
                            type="TSEdge")

        #print(f"TSEdge {oid} from {source} to {target} with label '{label}' successfully created.")
        # Mark the graph as updated
        self.set_updated()
        start_time=time_series.first_timestamp()
        # Update node degree time series
        #self._update_node_degree_time_series(tsedge.source, tsedge.target, operation='add', timestamp=start_time)

        return tsedge

    def add_subgraph(self, subgraph_id, label=None, properties=None, start_time=None, end_time=None,
                     node_filter=None, edge_filter=None):
        """
        Create a subgraph based on node and edge filters, and store it in the subgraphs dictionary.

        :param subgraph_id: Identifier for the subgraph.
        :param label: Label for the subgraph.
        :param properties: Dictionary of properties for the subgraph.
                           Each key is a property name, and each value is a tuple (value, property_type),
                           where property_type is 'static' or 'temporal'.
        :param start_time: Start time of the subgraph's validity.
        :param end_time: End time of the subgraph's validity.
        :param node_filter: Function that takes (node_id, data) and returns True if the node should be included.
        :param edge_filter: Function that takes (u, v, key, data) and returns True if the edge should be included.
        """
        # Check if subgraph_id already exists
        if subgraph_id in self.subgraphs:
            raise ValueError(f"Subgraph with ID '{subgraph_id}' already exists.")
        if end_time is None:
            end_time = FAR_FUTURE_DATE
        if start_time is None:
            start_time = datetime.now()

        # Create a subgraph view using NetworkX's subgraph_view
        subgraph_view = nx.subgraph_view(
            self.graph,
            filter_node=lambda n: node_filter(n, self.graph.nodes[n]) if node_filter else True,
            filter_edge=lambda u, v, k: edge_filter(u, v, k, self.graph.edges[u, v, k]) if edge_filter else True
        )

        # Create a Subgraph object with label and properties
        subgraph_obj = Subgraph(
            subgraph_id=subgraph_id,
            label=label or f"Subgraph {subgraph_id}",
            start_time=start_time,
            end_time=end_time
        )

        # Process properties
        if properties:
            for prop_name, prop_value in properties.items():
                if isinstance(prop_value, tuple) and len(prop_value) == 2:
                    value, prop_type = prop_value
                    if prop_type == 'static':
                        subgraph_obj.add_static_property(prop_name, value, self)
                    elif prop_type == 'temporal':
                        subgraph_obj.add_temporal_property(prop_name, value, self)
                    else:
                        raise ValueError(f"Invalid property type '{prop_type}' for property '{prop_name}'.")
                else:
                    # Default to static property if type is not specified
                    subgraph_obj.add_static_property(prop_name, prop_value, self)

        # Store the subgraph view and Subgraph object in the subgraphs dictionary
        self.subgraphs[subgraph_id] = {
            'view': subgraph_view,  # The NetworkX subgraph view
            'data': subgraph_obj,  # The Subgraph object with label and properties
            'created_at': datetime.now()  # Timestamp of creation
        }
        print(f"Subgraph '{subgraph_id}' created and stored at {start_time}.")
        return self.subgraphs[subgraph_id]



    def set_updated(self, value=True):
        self.updated = value





    def add_query(self, query):
        """
        Add a new query to the HyGraph instance.

        Parameters:
        - query (dict): A dictionary containing filters and configuration for time series creation.
        """
        self.query = query
        self.batch_process()
        print(f"Query added: {query}")
        # If needed, you can reinitialize the observer or notify it about the new query

    def set_query(self, query):
        """
        Sets or updates the query for the HyGraph instance.
        This query will be used by the observer to generate time series.
        """
        self.query = query
        print("Query has been set/updated.")

    def get_query(self):
        """
        Returns the current query  being used by the HyGraph instance.
        If an index is provided, return the specific query at that index.
        """
        return self.query

    def get_element(self, element_type, oid):
        if element_type == 'node':
            if oid not in self.graph.nodes:
                raise ValueError(f"Node with ID {oid} does not exist.")
            return self.graph.nodes[oid]['data']
        elif element_type == 'edge':
            for source, target, key in self.graph.edges(keys=True):
                if key == oid:
                    return self.graph.edges[source, target, key]['data']
            raise ValueError(f"Edge with ID {oid} does not exist.")
        elif element_type == 'subgraph':
            if oid not in self.subgraphs:
                raise ValueError(f"Subgraph with ID {oid} does not exist.")
            return self.subgraphs[oid]['data']


    def add_membership(self, element_id, timestamp, subgraph_ids,element_type):
        """
           Adds subgraph memberships to the element's TimeSeries at the given timestamp.
           """
        element = self.get_element(element_type, element_id)
        if not element:
            print(f"No element found with ID {element_id}")
            return

        timestamp = pd.Timestamp(timestamp).to_datetime64()  # Ensure consistent timestamp format

        if element.membership is None:
            # TimeSeries does not exist, create it
            tsid = self.id_generator.generate_timeseries_id()
            metadata = TimeSeriesMetadata(element_id, element_type, 'membership')
            # Initial memberships are the provided subgraph_ids
            updated_memberships = sorted(set(subgraph_ids))
            membership_string = ' '.join(updated_memberships)
            time_series = TimeSeries(tsid, [timestamp], ['membership'], [[membership_string]], metadata)
            self.time_series[tsid] = time_series
            element.membership = tsid
            #print(f"TimeSeries created for: {element_type} {element_id}: {time_series}")
        else:
            tsid = element.membership
            time_series = self.time_series[tsid]

            # Check if the timestamp already exists in the TimeSeries
            if timestamp in time_series.data.coords['time'].values:
                # Timestamp exists, get existing memberships at this timestamp
                existing_value = time_series.get_value_at_timestamp(timestamp)
            elif time_series.data.shape[0] > 0:
                # Get last known membership state
                existing_value = time_series.data.isel(time=-1).item()
            else:
                existing_value = ''

            existing_memberships = existing_value.split() if existing_value else []

            # Combine, deduplicate, and sort memberships
            updated_memberships = sorted(set(existing_memberships + subgraph_ids))
            membership_string = ' '.join(updated_memberships)

            if timestamp in time_series.data.coords['time'].values:
                # Update the value at the timestamp
                time_series.update_value_at_timestamp(timestamp, membership_string)
                print(f"Updated membership for {element_type} {element_id} at {timestamp}: add {subgraph_ids}")
            else:
                # Append new data point
                time_series.append_data(timestamp, membership_string)
                print(f"Appended new membership for {element_type} {element_id} at {timestamp}: add {subgraph_ids}")
    def remove_membership(self, element_id, timestamp, subgraph_ids,element_type):
        """
        Removes subgraph memberships from the element's TimeSeries at the given timestamp.
        """
        element = self.get_element(element_type, element_id)
        if not element:
            print(f"No element found with ID {element_id}")
            return

        if element.membership is None:
            print(f"No TimeSeries initialized for {element_type} {element_id}")

        else:
            tsid = element.membership
            time_series = self.time_series[tsid]
            # Get the last known membership state
            if time_series.data.shape[0] > 0:
                last_entry = time_series.data.isel(time=-1).item().split()
            else:
                last_entry = []
            # Remove specified subgraph IDs
            updated_membership = [sg for sg in last_entry if sg not in subgraph_ids]
            membership_string = ' '.join(updated_membership)

            time_series.append_data(timestamp, membership_string)
            print(f"Updated membership for {element_type} {element_id} at {timestamp}: remove {subgraph_ids}")

    def update_membership(self, element, timestamps, subgraph_ids,action):
        # Ensure that timestamps and subgraph_ids are aligned
        if len(timestamps) != len(subgraph_ids):
            raise ValueError("Timestamps and subgraph IDs arrays must have the same length.")

        tsid = element.membership.get('subgraph_membership') or self.id_generator.generate_timeseries_id()

        if tsid not in self.time_series:

            membership_series = TimeSeries(tsid, timestamps, ['membership'], [subgraph_ids])
            self.time_series[tsid] = membership_series
        else:
            membership_series = self.time_series[tsid]
            # Update existing time series based on action
            for timestamp, subgraph_id in zip(timestamps, subgraph_ids):
                if action == 'add':
                    # Use the existing append_data method to add the subgraph_id
                    membership_series.append_data(timestamp, subgraph_id)
                elif action == 'remove':
                    # Remove the subgraph ID for the given timestamp
                    index = membership_series.timestamps.index(
                        timestamp) if timestamp in membership_series.timestamps else None
                    if index is not None:
                        existing_data = membership_series.data[index]
                        if subgraph_id in existing_data:
                            existing_data.remove(subgraph_id)  # Remove subgraph ID if present
                            # Optionally: If you want to mark it as "removed" without deleting
                            if not existing_data:
                                existing_data.append(None)  # Placeholder to indicate removal

            element.membership['subgraph_membership'] = tsid
            print(
                f"Added subgraph membership to {type(element).__name__.lower()} {element.oid} with time series ID {tsid}.")


    def add_time_series(self, timestamps, variables, data, metadata=None):
        """
        Add a new time series to the hygraph.

        :param timestamps: List of timestamps for the time series
        :param variables: List of variables for the time series
        :param data: 2D array-like structure with data corresponding to timestamps and variables
        :param metadata: Optional metadata for the time series
        :return: The ID of the newly created time series
        """
        # Generate a new time series ID
        time_series_id = self.id_generator.generate_timeseries_id()
        # Create the time series
        if metadata ==None:
            metadata = TimeSeriesMetadata(-1)
        new_time_series = TimeSeries(time_series_id, timestamps, variables, data, metadata)

        # Store the time series in the hygraph
        self.time_series[time_series_id] = new_time_series

        #print(f"New time series added with ID {time_series_id}")
        return new_time_series


    #read
    def get_node_by_id(self, oid):
        """
       Retrieve a node from the graph using its Object ID (oid).

       :param oid: The Object ID of the node.
       :return: the node if found.
       :raises ValueError: If the node does not exist.
       """
        if oid not in self.graph.nodes:
            raise ValueError(f"Node with ID {oid} does not exist.")
        return self.graph.nodes[oid]

    def get_edge_by_id(self, oid):
        """
        Retrieve an edge from the graph using its Object ID (oid).

        :param oid: The Object ID of the edge.
        :return: A tuple (source, target, key, data) of the edge if found.
        :raises ValueError: If the edge does not exist.
        """
        for u, v, key, data in self.graph.edges(keys=True, data=True):
            if key == oid:
                return self.graph.edges(key)
        raise ValueError(f"Edge with ID {oid} does not exist.")

    def get_timeseries(self, tsid, display=False, limit=None, order='first'):
        """
        Retrieve and optionally display a time series by its ID.

        :param tsid: The ID of the time series to retrieve.
        :param display: If True, display the time series using its display method.
        :param limit: Limit the number of data points to display.
        :param order: 'first' or 'last' data points to display.
        :return: The TimeSeries object.
        """
        if tsid not in self.time_series:
            raise ValueError(f"Time series with ID {tsid} does not exist.")
        ts = self.time_series[tsid]
        if display:
            ts.display_time_series(limit=limit, order=order)
        return ts

    def get_subgraph_at(self, subgraph_id, timestamp):
        """
        Extracts the subgraph at a specific timestamp.

        :param subgraph_id: Identifier of the subgraph.
        :param timestamp: The timestamp at which to extract the subgraph.
        :return: A NetworkX graph representing the subgraph at the given time.
        """
        if subgraph_id not in self.subgraphs:
            raise ValueError(f"Subgraph with ID '{subgraph_id}' does not exist.")

        subgraph_obj = self.subgraphs[subgraph_id]

        # Extract nodes that are members of the subgraph at the given timestamp
        nodes_in_subgraph = []
        for node_id, data in self.graph.nodes(data=True):
            node = data.get('data')
            if node.membership:
                tsid = node.membership
                time_series = self.time_series[tsid]
                memberships = time_series.get_value_at_timestamp(timestamp)
                if memberships and subgraph_id in memberships:
                    nodes_in_subgraph.append(node_id)

        # Extract edges that are members of the subgraph at the given timestamp
        edges_in_subgraph = []
        for u, v, key, data in self.graph.edges(keys=True, data=True):
            edge = data.get('data')
            if edge.membership:
                tsid = edge.membership
                time_series = self.time_series[tsid]
                memberships = time_series.get_value_at_timestamp(timestamp)
                if memberships and subgraph_id in memberships:
                    edges_in_subgraph.append((u, v, key))

        # Create the subgraph
        #subgraph = self.graph.edge_subgraph(edges_in_subgraph).copy()
        subgraph = self.graph.subgraph(nodes_in_subgraph).copy()

        return subgraph

    def display_subgraph(self, subgraph_id, timestamp):
        """
        Display the subgraph with its associated nodes, edges, labels, start and end times, static and temporal properties.

        :param subgraph_id: Identifier of the subgraph to display.
        :param timestamp: The timestamp at which to display the subgraph.
        """
        subgraph_obj = self.subgraphs.get(subgraph_id)['data']
        if not subgraph_obj:
            print(f"Subgraph '{subgraph_id}' does not exist.")
            return

        # Display subgraph properties
        print(f"Subgraph '{subgraph_id}': Label: {subgraph_obj.label}")
        print(f"Start Time: {subgraph_obj.start_time}, End Time: {subgraph_obj.end_time}")
        # Display static properties
        if subgraph_obj.static_properties:
            print("Static Properties:")
            for prop_name, prop in subgraph_obj.static_properties.items():
                print(f"  {prop_name}: {prop.value}")
        # Display temporal properties
        if subgraph_obj.temporal_properties:
            print("Temporal Properties:")
            for prop_name, prop in subgraph_obj.temporal_properties.items():
                print(f"  {prop_name}: {prop}")
        print("---")

        # Get subgraph at the given timestamp
        subgraph_at_time = self.get_subgraph_at(subgraph_id, timestamp)

        # Display nodes
        print("Nodes in Subgraph:")
        for node_id, data in subgraph_at_time.nodes(data=True):
            node = data.get('data')
            print(f"Node ID: {node_id}, Label: {node.label}")

        # Display edges
        print("\nEdges in Subgraph:")
        for u, v, key, data in subgraph_at_time.edges(keys=True, data=True):
            edge = data.get('data')
            print(f"Edge ID: {key}, From: {u}, To: {v}, Label: {edge.label}")
            # Display edge's static properties

        print("---")

    def get_all_edges(self):
        """
        Return all edges in the graph.
        """
        return list(self.graph.edges(keys=True, data=True))

    def get_all_tsedges(self):
        """
        Return all TSEdge instances in the graph.
        """
        tsedges = []
        for u, v, key, data in self.graph.edges(keys=True, data=True):
            if data.get('type') == 'TSEdge':
                tsedges.append((u, v, key, data))
        return tsedges

    def get_all_pgedges(self):
        """
        Return all PGEdge instances in the graph.
        """
        pgedges = []
        for u, v, key, data in self.graph.edges(keys=True, data=True):
            if data.get('type') == 'PGEdge':
                pgedges.append((u, v, key, data))
        return pgedges

    def get_all_nodes(self):
        """
        Return all nodes in the graph.
        """
        return list(self.graph.nodes(data=True))

    def get_all_tsnodes(self):
        """
        Return all TSNode instances in the graph.
        """
        tsnodes = []
        for node_id, data in self.graph.nodes(data=True):
            if data.get('type') == 'TSNode':
                tsnodes.append((node_id, data))
        return tsnodes

    def get_all_pgnodes(self):
        """
        Return all PGNode instances in the graph.
        """
        pgnodes = []
        for node_id, data in self.graph.nodes(data=True):
            if data.get('type') == 'PGNode':
                pgnodes.append((node_id, data))
        return pgnodes

    def get_all_timeseries(self, display=False, limit=None, order='first'):
        """
        Retrieve and optionally display all time series in the HyGraph.

        :param display: If True, display each time series using its display method.
        :param limit: Limit the number of data points to display for each time series.
        :param order: 'first' or 'last' data points to display.
        :return: List of TimeSeries objects.
        """
        ts_list = list(self.time_series.values())
        if display:
            for ts in ts_list:
                ts.display_time_series(limit=limit, order=order)
        return ts_list

    def get_all_subgraphs(self):
        """
        Return all subgraphs stored in the HyGraph.
        """
        return list(self.subgraphs.values())

    def get_nodes_by_label(self, label):
        """
        Retrieve all nodes with the specified label.

        :param label: The label to match.
        :return: List of nodes matching the label.
        """
        nodes = []
        for node_id, data in self.graph.nodes(data=True):
            if data.get('label') == label:
                node_data = {key: value for key, value in data.items() if key != 'data'}
                nodes.append((node_id, node_data))
        return nodes

    def get_node_by_type(self, node_type):
        """
        Retrieve all nodes of a specific type.

        :param node_type: Type of the node to retrieve (e.g., 'TSNode' or 'PGNode').
        :return: List of nodes matching the specified type.
        """
        return [node for _, node in self.graph.nodes(data=True) if node['data'].get_type() == node_type]

    def get_edge_by_type(self, edge_type):
        """
        Retrieve all edges of a specific type.

        :param edge_type: Type of the edge to retrieve (e.g., 'PGEdge' or 'TSEdge').
        :return: List of edges matching the specified type.
        """
        return [edge_data['data'] for _, _, edge_data in self.graph.edges(data=True) if
                edge_data['data'].get_type() == edge_type]
    def get_nodes_by_static_property(self, property_name, condition):
        """
        Retrieve all nodes where the static property matches the given value.

        :param property_name: The name of the static property.
        :param value: The value to match.
        :return: List of nodes matching the static property condition.
        """
        nodes = []
        for node_id, data in self.graph.nodes(data=True):
            # Access the properties dictionary
            properties = data.get('properties', {})
            # Now retrieve the relevant temporal property
            static_prop = properties.get(property_name)

            if not static_prop or not isinstance(static_prop, StaticProperty):
                continue
            if condition(static_prop):
                node_data = {key: value for key, value in data.items() if key != 'data'}
                nodes.append((node_id, node_data))
        return nodes

    def get_nodes_by_temporal_property(self, property_name, condition):
        """
        Retrieve all nodes where the temporal property satisfies the given condition.

        :param property_name: The name of the temporal property.
        :param condition: A function that takes a TimeSeries object and returns True if the condition is met.
        :return: List of nodes matching the temporal property condition.
        """
        nodes = []
        for node_id, data in self.graph.nodes(data=True):
            # Access the properties dictionary
            properties = data.get('properties', {})
            # Now retrieve the relevant temporal property
            temporal_prop = properties.get(property_name)

            if temporal_prop and  isinstance(temporal_prop, TemporalProperty):
                ts = temporal_prop.get_time_series()
                if condition(ts):
                    node_data = {key: value for key, value in data.items() if key != 'data'}
                    nodes.append((node_id, node_data))
        return nodes

    def get_edges_by_label(self, label):
        """
        Retrieve all edges with the specified label.

        :param label: The label to match.
        :return: List of edges matching the label.
        """
        edges = []
        for u, v, key, data in self.graph.edges(keys=True, data=True):
            if data.get('label') == label:
                edges.append((u, v, key, data))
        return edges

    def get_subgraphs_by_label(self, label):
        """
        Retrieve all subgraphs with the specified label.

        :param label: The label to match.
        :return: List of subgraphs matching the label.
        """
        subgraphs = []
        for subgraph_id, subgraph_data in self.subgraphs.items():
            subgraph = subgraph_data['data']
            if subgraph.label == label:
                subgraphs.append(subgraph)
        return subgraphs

    def get_subgraphs_by_static_property(self, property_name, condition):
        """
        Retrieve all subgraphs where the static property satisfies the given condition.

        :param property_name: The name of the static property.
        :param condition: A function that takes a property value and returns True if the condition is met.
        :return: List of subgraphs matching the static property condition.
        """
        subgraphs = []
        for subgraph_id, subgraph_data in self.subgraphs.items():
            subgraph = subgraph_data['data']
            static_prop = subgraph.static_properties.get(property_name)
            if static_prop and condition(static_prop.value):
                subgraphs.append(subgraph)
        return subgraphs

    #deletion

    def delete_node(self, oid, end_time=None):
        if oid not in self.graph.nodes:
            raise ValueError(f"Node with ID {oid} does not exist.")

        node = self.get_element('node', oid)
        end_time = end_time or datetime.now()
        node.end_time = end_time
        print(f"Node {oid} logically deleted with end_time {node.end_time}.")
        # Optionally, logically delete connected edges
        for neighbor in self.graph.neighbors(oid):
            for key in list(self.graph[oid][neighbor]):
                self.delete_edge(key, end_time=end_time)

        # Notify observers
        self.set_updated()
        node.notify()

    def delete_edge(self, oid, end_time=None):
        edge = None
        for u, v, key in self.graph.edges(keys=True):
            if key == oid:
                edge = self.graph.edges[u, v, key]['data']
                break

        if edge is None:
            raise ValueError(f"Edge with ID {oid} does not exist.")

        end_time = end_time or datetime.now()
        edge.end_time = end_time
        print(f"Edge {oid} logically deleted with end_time {edge.end_time}.")

        # Notify observers
        self.set_updated()
        edge.notify()
        #self._update_node_degree_time_series(edge.source, edge.target, operation='remove', timestamp=end_time)


    def delete_subgraph(self, subgraph_id, end_time=None):
        """
        Logically delete a subgraph by setting its end_time.

        :param subgraph_id: The ID of the subgraph to be logically deleted.
        :param end_time: The end_time to set. If None, uses datetime.now().
        """
        if subgraph_id in self.subgraphs:
            subgraph_data = self.subgraphs[subgraph_id]
            subgraph_obj = subgraph_data.get('data')
            if subgraph_obj:
                # Set the end_time
                subgraph_obj.end_time = end_time or datetime.now()
                print(f"Subgraph '{subgraph_id}' logically deleted with end_time {subgraph_obj.end_time}.")
            else:
                print(f"Subgraph '{subgraph_id}' has no associated data object.")
        else:
            print(f"Subgraph '{subgraph_id}' does not exist.")

    
    def append_time_series(self, tsid, date, value):
        if tsid in self.time_series:
            self.time_series[tsid].append_data(date, value)
        else:
            raise ValueError(f"Time series with ID {tsid} does not exist.")

    def find_path(self, source_id, target_id, method='dijkstra', weight_property=None):
        """
        Finds a path between two nodes in HyGraph using the specified method.

        :param hygraph: HyGraph instance.
        :param source_id: ID of the source node.
        :param target_id: ID of the target node.
        :param method: Pathfinding method ('bfs', 'dfs', 'astar').
        :param weight_property: Property for edge weights (for 'astar' only).
        :param max_length: Maximum path length (cutoff for BFS/DFS).
        :return: Dictionary with path nodes and edges.
        """
        # Choose pathfinding algorithm
        if method == 'astar':
            if not weight_property:
                raise ValueError("A* method requires a weight property.")
            path_nodes = nx.astar_path(self.graph, source=source_id, target=target_id, weight=weight_property)
        elif method == 'dijkstra':
            path_nodes = nx.shortest_path(self.graph, source=source_id, target=target_id, method="dijkstra")
        else:
            raise ValueError("Method must be 'bfs', 'dfs', or 'astar'.")

        # Collect nodes and edges along the path
        if path_nodes:
            path_edges = [
                self.graph.get_edge_data(u, v)['data']
                for u, v in path_nodes
            ]
            path_data = {
                "nodes": [self.graph.nodes[n]['data'] for n in [source_id] + [v for u, v in path_nodes]],
                "edges": path_edges
            }
            return path_data
        else:
            return {"nodes": [], "edges": []}

    def get_node_degree_over_time(self, node_id, degree_type='both', return_type='history'):
        """
        Retrieve the node degree over time.

        :param node_id: ID of the node
        :param degree_type: 'in', 'out', or 'both'
        :param return_type: 'history' or 'current'
        :return: Degree time series or current degree value
        """
        node = self.graph.nodes[node_id]['metadata']
        degrees = []

        if degree_type in ['in', 'both']:
            in_degree_ts = node['in_degree']  # Access in_degree from metadata
            degrees.append(in_degree_ts)

        if degree_type in ['out', 'both']:
            out_degree_ts = node['out_degree']  # Access out_degree from metadata
            degrees.append(out_degree_ts)

        if degrees:
            if return_type == 'history':
                # Merge the time series
                combined_ts = TimeSeries.aggregate_multiple(degrees, method='sum')
                return combined_ts
            elif return_type == 'current':
                # Return the latest degree value
                current_degree = sum(ts.last_value()[1] for ts in degrees)
                return current_degree
        else:
            return None  # No degree information available


    def batch_process(self):
        """Batch processing function to process all nodes and edges."""
        print("Starting batch processing...")
        self.create_time_series_from_graph(self.query)  # Update time series for all nodes and edges
        print("Batch processing completed.")

    def display(self):
        print("Nodes:")
        for node_id, data in self.graph.nodes(data=True):
            print(f"Node {node_id}: {data}")

        print("\nEdges:")
        for source, target, key, data in self.graph.edges(keys=True, data=True):
            print(f"Edge {key} from {source} to {target}: {data['data']}")

        print("\nSubgraphs:")
        for subgraph_id, data in self.subgraphs.items():
            print(f"Subgraph {subgraph_id}: {data}")

        print("\nTime Series:", len(self.time_series))
        for tsid, ts in self.time_series.items():
            print(f"Time Series {tsid}: {ts.metadata.owner_id}")
            variables = [str(var) for var in ts.data.coords['variable'].values]
            print(f"Variables: {', '.join(variables)}")
            ts_df = ts.data.to_dataframe('value').reset_index()
            grouped = ts_df.groupby('time')
            for time, group in grouped:
                values = [f" {row['value']}" for idx, row in group.iterrows()]
                row_str = ", ".join(values)
                print(f"{time}, {row_str}")
        self.set_updated(False)  # Reset the flag after displaying

#utility functions

    #functions to calculate node degree over time
    def _update_node_degree_time_series(self, source_id, target_id, operation, timestamp):
        """
        Update the node degree time series for source and target nodes.

        :param source_id: ID of the source node
        :param target_id: ID of the target node
        :param operation: 'add' or 'remove'
        :param timestamp: The time when the edge was added or removed
        """
        source_node = self.graph.nodes[source_id]['metadata']
        target_node = self.graph.nodes[target_id]['metadata']

        # Update out-degree for source node
        self._update_degree_time_series_for_node(source_node, degree_type='out', operation=operation,
                                                 timestamp=timestamp)

        # Update in-degree for target node
        self._update_degree_time_series_for_node(target_node, degree_type='in', operation=operation,
                                                 timestamp=timestamp)

    def _update_degree_time_series_for_node(self, node, degree_type, operation, timestamp):
        """
        Update the degree time series for a node.

        :param node: The node object
        :param degree_type: 'in' or 'out'
        :param operation: 'add' or 'remove'
        :param timestamp: The time when the edge was added or removed
        """
        property_name = f"{degree_type}_degree"
        tsid = None
        last_degree = 0
        # Check if the node already has a degree time series
        if property_name in node:
            tsid = node[property_name].tsid
            time_series = self.time_series[tsid]
            last_degree = time_series.data.isel(time=-1).values.item()
            # Update the degree based on the operation
            new_degree = last_degree + 1 if operation == 'add' else max(0, last_degree - 1)
            # Update the existing value at the timestamp or append a new entry
            if time_series.has_timestamp(timestamp):
                time_series.update_value_at_timestamp(timestamp, new_degree)
            else:
                # Append the new degree value with the timestamp
                time_series.append_data(timestamp, new_degree)
        else:
            # Create a new time series for the degree
            tsid = self.id_generator.generate_timeseries_id()
            metadata = TimeSeriesMetadata(owner_id=node.oid, element_type='node')
            # Set the initial degree based on the operation
            new_degree = 1 if operation == 'add' else 0
            # Wrap the data in a compatible format for xarray
            data = np.array([[new_degree]])  # Ensure data has dimensions [time, variable]
            time_index = pd.DatetimeIndex([timestamp])

            # Create the TimeSeries instance with correctly shaped data
            time_series = TimeSeries(tsid, time_index, [property_name], data, metadata)
            self.time_series[tsid] = time_series
            node[property_name] = time_series  # Store directly in metadata

    #hybrid operators


    #generate timeseries from graph
    def create_time_series_from_graph(self, query):
        """
        Create time series data for graph elements (nodes, edges, or subgraphs) based on a provided query.

        Query dictionary supports:
          - 'element_type': 'node', 'edge', or 'subgraph'
          - 'node_filter': function(node_data) -> bool
          - 'edge_filter': function(edge_data) -> bool
          - 'subgraph_id' or 'subgraph_label' or 'subgraph_filter' (for subgraphs)
          - 'aggregation': Boolean flag; for edge elements, if True the system aggregates edges into a super edge.
          - 'target': For node TS extraction, indicates which edge direction to consider ('source', 'target', or 'both').
          - 'node_settings': Dict (e.g., {'include_edges': [list of edge labels], 'aggregate_edges': True})
          - 'time_series_config': Dict with keys:
                'start_date'        : datetime for TS range (if not using actual timestamps)
                'end_date'          : datetime for TS range (if not using actual timestamps)
                'freq'              : Frequency string (e.g., 'D' for daily, 'M' for monthly)
                'attribute'         : Base name for the dynamic property (e.g., 'number_connections')
                'aggregate_function': Function with signature (hygraph, element_type, oid, attribute, dt) -> numeric value
                'direction'         : For edge aggregation ('in', 'out', or 'both')
                'use_actual_timestamps': Boolean flag; if True, TS is built using collected event timestamps.
        """
        element_type = query.get('element_type', 'node')
        ts_config = query.get('time_series_config', {})

        if element_type == 'node':
            self._process_nodes_for_ts(query, ts_config)
        elif element_type == 'subgraph':
            self._process_subgraphs_for_ts(query, ts_config)
        elif element_type == 'edge':
            self._process_edges_for_ts(query, ts_config)
        else:
            raise ValueError(f"Unsupported element type for time series generation: {element_type}")

    def filter_nodes(self, node_filter, edge_filter):
        """
        Filters nodes from the HyGraph based on a node_filter and an optional edge_filter.

        Parameters:
          - node_filter: Function that takes node data (the dictionary from self.graph.nodes) and returns True if the node should be included.
          - edge_filter: (Optional) Function that takes an edge dictionary and returns True if the edge satisfies the criteria.

        Returns:
          - A list of dictionaries containing the node data under the key 'data'.
        """
        filtered_nodes = []
        for node_id, data in self.graph.nodes(data=True):
            # data is expected to have a 'data' field holding the actual node object.
            if node_filter(data):
                if edge_filter:
                    # Check if at least one incident edge satisfies the edge_filter.
                    incident_edges = self.graph.edges(node_id, keys=True, data=True)
                    if any(edge_filter({'data': edata}) for _, _, _, edata in incident_edges):
                        filtered_nodes.append({'data': data['data']})
                else:
                    filtered_nodes.append({'data': data['data']})
        return filtered_nodes

    def _aggregate_edges_for_node(self, node_obj, target, node_settings, ts_config):
        """
        Aggregate incident edges of a node based on the provided settings and aggregator function.

        Parameters:
          - node_obj: The node object (e.g., a PGNode) from which to aggregate edges.
          - target: String indicating which edges to consider ('source', 'target', or 'both').
          - node_settings: Dictionary that contain 'include_edges' (a list of edge labels to include).
          - ts_config: Time series configuration dictionary; must contain 'aggregate_function' and 'attribute'.

        Returns:
          - A numeric value (or tuple) that is the aggregation result of the eligible edges.
        """
        include_edges = node_settings.get('include_edges', [])
        node_id = node_obj.oid

        # Select edges based on target.
        if target == 'source':
            edges = self.graph.out_edges(node_id, keys=True, data=True)
        elif target == 'target':
            edges = self.graph.in_edges(node_id, keys=True, data=True)
        else:
            edges = self.graph.edges(node_id, keys=True, data=True)

        aggregator = ts_config['aggregate_function']
        base_attribute = ts_config.get('attribute', 'default_attribute')

        # Initialize aggregated value by sampling an eligible edge.
        agg_value = None
        for u, v, key, data in edges:
            edge_obj = data['data']
            if include_edges and edge_obj.label not in include_edges:
                continue
            sample = aggregator(self, 'edge', key, base_attribute, datetime.now())
            if isinstance(sample, tuple):
                agg_value = tuple(0 for _ in sample)
            else:
                agg_value = 0
            break  # Initialization done.

        if agg_value is None:
            # No eligible edges found.
            return 0

        # Sum aggregator values over all eligible edges.
        for u, v, key, data in edges:
            edge_obj = data['data']
            if include_edges and edge_obj.label not in include_edges:
                continue
            edge_val = aggregator(self, 'edge', key, base_attribute, datetime.now())
            if isinstance(edge_val, tuple):
                agg_value = tuple(sum(x) for x in zip(agg_value, edge_val))
            else:
                agg_value += edge_val
        return agg_value
    def _process_nodes_for_ts(self, query, ts_config):
        node_filter = query.get('node_filter', lambda node: True)
        edge_filter = query.get('edge_filter', None)
        selected_nodes = self.filter_nodes(node_filter, edge_filter)
        node_settings = query.get('node_settings', {})
        aggregate_edges_flag = node_settings.get('aggregate_edges', False)
        target = query.get('target', None)
        base_attribute = ts_config.get('attribute', 'default_attribute')

        for node_data in selected_nodes:
            node_obj = node_data['data']
            existing_ts_id = node_obj.properties.get(base_attribute)
            if existing_ts_id and existing_ts_id in self.time_series:
                # TS already exists – append a new value at the current timestamp.
                current_value = ts_config['aggregate_function'](self, 'node', node_obj.oid, base_attribute,
                                                                datetime.now())
                self.append_time_series(existing_ts_id, datetime.now(), current_value)
            else:
                if aggregate_edges_flag and target:
                    # If aggregation is enabled, compute an aggregated value from incident edges.
                    agg_val = self._aggregate_edges_for_node(node_obj, target, node_settings, ts_config)
                    new_ts = self.add_time_series([datetime.now()], [base_attribute], [[agg_val]])
                    node_obj.add_temporal_property(base_attribute, new_ts, self)
                else:
                    # Otherwise, compute a time series over the specified time range.
                    self._compute_ts_for_element(node_obj, ts_config, element_type='node')

    def _process_subgraphs_for_ts(self, query, ts_config):
        subgraph_id = query.get('subgraph_id')
        subgraph_label = query.get('subgraph_label')
        subgraph_filter = query.get('subgraph_filter', lambda s: True)

        if subgraph_id:
            subgraphs = [self.get_element('subgraph', subgraph_id)]
        elif subgraph_label:
            subgraphs = [s['data'] for s in self.subgraphs.values() if s['data'].label == subgraph_label]
        else:
            subgraphs = [s['data'] for s in self.subgraphs.values()]

        filtered_subgraphs = [sg for sg in subgraphs if subgraph_filter(sg)]
        if not filtered_subgraphs:
            print("No subgraphs matched the provided filter criteria.")
            return

        for subgraph_obj in filtered_subgraphs:
            self._compute_ts_for_element(subgraph_obj, ts_config, element_type='subgraph')

    def _process_edges_for_ts(self, query, ts_config):
        """
        Process edge elements for time series generation.
        If 'aggregation' is True, then edges are grouped by (source, target, label)
        and a "super edge" is created. The aggregated TS is attached as a dynamic property
        on the super edge.
        """
        edge_filter = query.get('edge_filter', lambda edge: True)
        aggregation_flag = query.get('aggregation', False)
        selected_edges = [(u, v, key, data) for u, v, key, data in self.graph.edges(keys=True, data=True)
                          if edge_filter(data)]

        if aggregation_flag:
            # Group edges by a grouping key; default group-by: (source, target, label)
            group_by_keys = query.get('edge_settings', {}).get('group_by', ['source', 'target', 'label'])
            groups = {}
            for u, v, key, data in selected_edges:
                edge_obj = data['data']
                # For now, we assume grouping by source, target, and label.
                group_key = (u, v, edge_obj.label)
                groups.setdefault(group_key, []).append((u, v, key, data))

            for group_key, edges in groups.items():
                # Compute aggregated time series for the group.
                new_ts = self._compute_ts_for_edge_group(edges, ts_config)
                if new_ts is None:
                    continue
                # Create a new super edge.
                new_oid = f"super_{group_key[0]}_{group_key[1]}_{group_key[2]}"
                # For start_time, we take the minimum start time among edges.
                edge_start_times = [data.get('start_time') for _, _, _, data in edges if
                                    data.get('start_time') is not None]
                if edge_start_times:
                    start_time = min(edge_start_times)
                else:
                    start_time = datetime.now()
                new_edge = self.add_pgedge(new_oid, group_key[0], group_key[1], group_key[2], start_time, None, properties={})
                # Attach the aggregated TS as a dynamic property on the super edge.
                new_edge.add_temporal_property( ts_config['attribute'], new_ts, self)

        else:
            # Without aggregation: Process each edge individually.
            for u, v, key, data in selected_edges:
                edge_obj = data['data']
                self._compute_ts_for_element(edge_obj, ts_config, element_type='edge')

    def _compute_ts_for_element(self, element_obj, ts_config, element_type):
        """
        Compute a time series for a given element (node, edge, or subgraph) using either
        actual event timestamps or a generated date range.
        """

        base_attribute = ts_config.get('attribute', 'default_attribute')
        agg_func = ts_config.get('aggregate_function')

        # Decide on timestamps: if use_actual_timestamps is True, collect them from events.
        if ts_config.get('use_actual_timestamps', False):
            timestamps = self._collect_timestamps_for_element(element_obj, element_type)
            if not timestamps:
                # Fallback to generated range.
                start_date = ts_config.get('start_date', datetime.now())
                end_date = ts_config.get('end_date', datetime.now())
                freq = ts_config.get('freq', 'D')
                timestamps = list(pd.date_range(start=start_date, end=end_date, freq=freq))
        else:
            start_date = ts_config.get('start_date', datetime.now())
            end_date = ts_config.get('end_date', datetime.now())
            freq = ts_config.get('freq', 'D')
            timestamps = list(pd.date_range(start=start_date, end=end_date, freq=freq))

        values = []
        # Extract the element's global start_time and end_time
        st = getattr(element_obj, 'start_time', None)
        et = getattr(element_obj, 'end_time', None)

        for dt in timestamps:
            val = 0
            # (A) Skip if dt not in the element’s validity interval
            if st and dt >= st and et and dt < et:
                # (B) If active, call aggregator
                val = agg_func(self, element_type, element_obj.oid, base_attribute, dt)

            values.append([val])

        if values:
            new_ts = self.add_time_series(timestamps, [base_attribute], values)
            element_obj.add_temporal_property(base_attribute,new_ts,self)
        else:
            print(f"No data computed for {element_obj.oid} in the specified range.")

    def _collect_timestamps_for_element(self, element_obj, element_type):
        """
        Collect actual timestamps from events associated with the element.
        For a node, this includes the node's start/end times and the start times of incident edges.
        For an edge, it could include its own start time.
        For subgraphs, combine timestamps from all contained nodes and edges.
        """
        timestamps = set()
        if element_type == 'node':
            timestamps.add(element_obj.start_time)
            timestamps.add(element_obj.end_time)
            for u, v, key, data in self.graph.edges(element_obj.oid, keys=True, data=True):
                st = data.get('start_time', None)
                if st:
                    timestamps.add(st)
        elif element_type == 'edge':
            st = element_obj.start_time if hasattr(element_obj, 'start_time') else None
            if st:
                timestamps.add(st)
        elif element_type == 'subgraph':
            for node_id, data in self.graph.nodes(data=True):
                node_obj = data.get('data')
                if node_obj:
                    timestamps.add(node_obj.start_time)
                    timestamps.add(node_obj.end_time)
            for u, v, key, data in self.graph.edges(keys=True, data=True):
                st = data.get('start_time', None)
                if st:
                    timestamps.add(st)
        return sorted(list(timestamps))

    def _compute_ts_for_edge_group(self, edges, ts_config):
        """
        For a group of edges (assumed to share source, target, and label), compute a time series
        by aggregating the aggregator function's output across the group.
        """

        base_attribute = ts_config.get('attribute', 'default_attribute')
        agg_func = ts_config.get('aggregate_function')

        # Decide on timestamps: if use_actual_timestamps, collect from all edges.
        if ts_config.get('use_actual_timestamps', False):
            timestamps = set()
            for u, v, key, data in edges:
                st = data.get('start_time', None)
                if st:
                    timestamps.add(st)
            timestamps = sorted(list(timestamps))
            if not timestamps:
                start_date = ts_config.get('start_date', datetime.now())
                end_date = ts_config.get('end_date', datetime.now())
                freq = ts_config.get('freq', 'D')
                timestamps = list(pd.date_range(start=start_date, end=end_date, freq=freq))
        else:
            start_date = ts_config.get('start_date', datetime.now())
            end_date = ts_config.get('end_date', datetime.now())
            freq = ts_config.get('freq', 'D')
            timestamps = list(pd.date_range(start=start_date, end=end_date, freq=freq))

        values = []
        for dt in timestamps:
            group_val = 0
            for u, v, key, data in edges:
                edge_obj = data['data']
                st = getattr(edge_obj, 'start_time', None)
                et = getattr(edge_obj, 'end_time', None)
                if st and dt>=st and et and dt<et:
                    group_val += agg_func(self, 'edge', key, base_attribute, dt)
            values.append([group_val])

        if values:
            return self.add_time_series(timestamps, [base_attribute], values)
        else:
            print("No aggregated data computed for the edge group.")
            return None


    #DFS with ts similarity
    def find_nearest_node_with_similar_timeseries(
        hygraph,
        start_node_id,
        similarity_func_name,
        threshold,
        property_name=None,
        inverse_similarity=False,
        variable_name=None
    ):
        """
        Find the nearest node to the start_node_id that has similar time series behavior.

        Args:
            hygraph (HyGraph): The HyGraph instance containing the graph and time series data.
            start_node_id (str): The starting node's ID.
            similarity_func_name (str): Name of the similarity function to use.
            threshold (float): The similarity threshold to determine if two time series are similar.
            property_name (str, optional): The name of the temporal property containing the time series ID for PGNode.
            inverse_similarity (bool, optional): If True, uses the inverse of the similarity measure.
            variable_name (str, optional): The name of the variable in the time series to compare.

        Returns:
            node_id or None: The nearest node with similar time series behavior, or None if not found.
        """

        # Begin BFS traversal
        visited = set()
        queue = deque()
        queue.append(start_node_id)
        visited.add(start_node_id)

        # Get the time series for the start node
        start_node_data = hygraph.graph.nodes[start_node_id]
        start_node_obj = start_node_data.get('data')  # Get the node object (PGNode or TSNode)

        # Retrieve the start node's time series
        start_ts = None

        if property_name:
            # Get time series from temporal_properties of the node
            if hasattr(start_node_obj, 'temporal_properties'):
                temporal_properties = start_node_obj.temporal_properties
                if property_name in temporal_properties:
                    temporal_prop = temporal_properties[property_name]
                    time_series_id = temporal_prop.time_series_id
                    start_ts = hygraph.time_series[time_series_id]
                else:
                    raise ValueError(f"Property '{property_name}' not found in node {start_node_id}'s temporal properties.")
            else:
                raise ValueError(f"Node {start_node_id} does not have temporal properties.")
        else:
            # Node must be TSNode or TSEdge, directly access the time series
            if isinstance(start_node_obj, (TSNode, TSEdge)):
                start_ts = start_node_obj.series  # This is a TimeSeries object
            else:
                raise ValueError(f"Property name not provided, and node {start_node_id} is not a TSNode or TSEdge.")

        # Map similarity function names to TimeSeries methods
        similarity_methods = {
            'euclidean_distance': start_ts.euclidean_distance,
            'correlation_coefficient': start_ts.correlation_coefficient,
            'cosine_similarity': start_ts.cosine_similarity,
            'manhattan_distance': start_ts.manhattan_distance,
            'dynamic_time_warping': start_ts.dynamic_time_warping
        }

        if similarity_func_name not in similarity_methods:
            raise ValueError(f"Similarity function '{similarity_func_name}' not recognized.")

        similarity_method = similarity_methods[similarity_func_name]

        # Now perform BFS traversal
        while queue:
            current_node_id = queue.popleft()
            if current_node_id == start_node_id:
                continue  # Already processed

            current_node_data = hygraph.graph.nodes[current_node_id]
            current_node_obj = current_node_data.get('data')

            current_ts = None

            if property_name:
                # Get time series from temporal_properties
                if hasattr(current_node_obj, 'temporal_properties'):
                    temporal_properties = current_node_obj.temporal_properties
                    if property_name in temporal_properties:
                        temporal_prop = temporal_properties[property_name]
                        time_series_id = temporal_prop.time_series_id
                        current_ts = hygraph.time_series[time_series_id]
                    else:
                        continue  # Property not found, skip this node
                else:
                    continue  # Node does not have temporal properties
            else:
                # Node must be TSNode or TSEdge
                if isinstance(current_node_obj, (TSNode, TSEdge)):
                    current_ts = current_node_obj.series
                else:
                    continue  # Not a TSNode or TSEdge, skip

            # Now compute similarity
            if start_ts is not None and current_ts is not None:
                try:
                    similarity = similarity_method(current_ts)

                    if inverse_similarity:
                        similarity = -similarity  # Assuming higher negative values mean less similar

                    if similarity >= threshold:
                        return current_node_id  # Found a similar node
                except ValueError as e:
                    # Handle cases where time series lengths do not match or other errors
                    print(f"Skipping node {current_node_id} due to error: {e}")
                    continue

            # Add neighbors to queue
            for neighbor in hygraph.graph.neighbors(current_node_id):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return None  # No similar node found



class HyGraphQuery:
    def __init__(self, hygraph):
        self.hygraph = hygraph
        self.node_matches = {}
        self.edge_matches = {}
        self.patterns = []
        self.conditions = []
        self.return_elements = []
        self.groupings = []
        self.aggregations = []
        self.orderings = []
        self.limit_count = None
        self.distinct_flag = False
        self.result_type = 'elements'
        self.current_alias = None
        self.subquery_results = None

        # Indexing structures for optimization
        self.node_index = defaultdict(set)  # {label: set(node_ids)}
        self.edge_index = defaultdict(set)  # {label: set(edge_keys)}

        # Build indices
        self._build_indices()

    def _build_indices(self):
        # Build node index based on labels
        for node_id, data in self.hygraph.graph.nodes(data=True):
            label = data.get('label')
            if label:
                self.node_index[label].add(node_id)

        # Build edge index based on labels
        for u, v, key, data in self.hygraph.graph.edges(keys=True, data=True):
            label = data.get('label')
            if label:
                self.edge_index[label].add((u, v, key))

    def match_node(self, alias, label=None, node_type=None, node_id=None):
        self.current_alias = alias
        self.node_matches[alias] = {
            'label': label,
            'type': node_type,
            'id': node_id,
            'matches': None  # To store matched nodes
        }
        return self

    def match_edge(self, alias, label=None, edge_type=None,edge_id=None):
        self.current_alias = alias
        self.edge_matches[alias] = {
            'label': label,
            'type': edge_type,
             'id': edge_id,
            'matches': None  # To store matched edges
        }
        return self

    def connect(self, source_alias, edge_alias, target_alias):
        # Validate aliases
        if source_alias not in self.node_matches:
            raise ValueError(f"Source alias '{source_alias}' not defined.")
        if edge_alias not in self.edge_matches:
            raise ValueError(f"Edge alias '{edge_alias}' not defined.")
        if target_alias not in self.node_matches:
            raise ValueError(f"Target alias '{target_alias}' not defined.")

        self.patterns.append({
            'source': source_alias,
            'edge': edge_alias,
            'target': target_alias
        })
        return self

    def where(self, condition):
        if self.current_alias is None:
            raise ValueError("No current alias to apply the condition to.")
        self.conditions.append({
            'alias': self.current_alias,
            'condition': condition
        })
        return self

    def group_by(self, *groupings):
        """
        Groups results based on provided aliases or functions that extract grouping keys from the result.

        :param groupings: Aliases (strings) or functions that take a result dictionary and return a grouping key.
        """
        for grouping in groupings:
            if callable(grouping):
                # If grouping is a function, add it directly
                self.groupings.append(grouping)
            elif isinstance(grouping, str):
                # If grouping is an alias, add the alias directly to self.groupings
                self.groupings.append(grouping)
            else:
                raise ValueError("group_by accepts either alias strings or functions.")
        return self

    def aggregate(self, alias, property_name, method='sum', direction='both', fill_value=0):
        """
        Perform aggregation over nodes or edges, handling both static and time series properties.

        :param alias: Alias of the graph element to aggregate.
        :param property_name: The property name to aggregate.
        :param method: Aggregation method ('sum', 'mean', 'min', 'max').
        :param direction: Edge direction to consider for edges ('in', 'out', 'both').
        :param fill_value: Value to fill for missing timestamps in time series.
        """
        self.aggregations.append({
            'alias': alias,
            'property_name': property_name,
            'method': method,
            'direction': direction,
            'fill_value': fill_value
        })
        return self

    def order_by(self, key, ascending=True):
        self.orderings.append({
            'key': key,
            'ascending': ascending
        })
        return self

    def limit(self, count):
        self.limit_count = count
        return self

    def distinct(self):
        self.distinct_flag = True
        return self

    def return_(self, **aliases):
        self.return_elements.extend(aliases.items())
        return self

    def result_as(self, result_type):
        self.result_type = result_type
        return self

    def subquery(self, subquery_func):
        """
        Use the result of a subquery in the current query.
        The subquery_func should return a list of node or edge instances.
        """
        subquery_results = subquery_func(HyGraphQuery(self.hygraph).execute())
        self.subquery_results = subquery_results
        return self



    def execute(self):
        # Match nodes and edges
        self._match_nodes()
        self._match_edges()

        # Apply conditions
        self._apply_conditions()

        # Match patterns
        if self.patterns:
            results = self._match_patterns()
        else:
            results = self._combine_matches()

        # Apply grouping and aggregations
        results = self._apply_aggregations(results)

        # Apply distinct
        results = self._apply_distinct(results)

        # Apply ordering
        results = self._apply_ordering(results)

        # Apply limit
        results = self._apply_limit(results)

        # Format results
        formatted_results = self._format_results(results)

        return formatted_results

    def _match_nodes(self):
        for alias, criteria in self.node_matches.items():
            label = criteria['label']
            node_type = criteria['type']
            node_id = criteria['id']
            matches = []
            # If node_id is specified, match this node directly and skip further processing
            if node_id:
                node_data = self.hygraph.graph.nodes[node_id]
                node_obj = node_data.get('data')  # PGNode, TSNode, etc.
                matches.append(node_obj)
                criteria['matches'] = matches
                continue

            # Get candidate nodes from the index if label is provided
            if label:
                candidate_node_ids = self.node_index.get(label, set())
            else:
                candidate_node_ids = set(self.hygraph.graph.nodes())

            # Filter by node_type and collect node objects
            matches = []
            for node_id in candidate_node_ids:
                data = self.hygraph.graph.nodes[node_id]
                if node_type and data.get('type') != node_type:
                    continue
                node_obj = data.get('data')  # This should be the PGNode, TSNode, etc.
                if node_obj:
                    matches.append(node_obj)
            criteria['matches'] = matches

    def _match_edges(self):
        for alias, criteria in self.edge_matches.items():
            label = criteria['label']
            edge_type = criteria['type']
            edge_id = criteria['id']
            # Get candidate edges from index
            matches = []
            if edge_id:
                for u, v, key, data in self.hygraph.graph.edges(keys=True, data=True):
                    edge_obj = data.get('data')  # Should be the PGEdge, TSEdge, etc.
                    if data.get('oid') == edge_id and edge_obj:
                        matches.append(edge_obj)
                criteria['matches'] = matches
                continue  # Skip further processing if edge_id is provided

            if label:
                candidate_edges = self.edge_index.get(label, set())
            else:
                candidate_edges = set(self.hygraph.graph.edges(keys=True))

            # Filter by edge_type and collect edge objects

            for u, v, key in candidate_edges:
                data = self.hygraph.graph.edges[u, v, key]
                if edge_type and data.get('type') != edge_type:
                    continue
                edge_obj = data.get('data')  # This should be the PGEdge, TSEdge, etc.
                if edge_obj:
                    matches.append(edge_obj)
            criteria['matches'] = matches

    def _apply_conditions(self):
        # Apply conditions to matched nodes and edges
        for condition in self.conditions:
            alias = condition['alias']
            cond_func = condition['condition']

            if alias in self.node_matches:
                matches = self.node_matches[alias]['matches']
                filtered_matches = [node for node in matches if cond_func(node)]
                self.node_matches[alias]['matches'] = filtered_matches
            elif alias in self.edge_matches:
                matches = self.edge_matches[alias]['matches']
                filtered_matches = [edge for edge in matches if cond_func(edge)]
                self.edge_matches[alias]['matches'] = filtered_matches
            else:
                raise ValueError(f"Alias '{alias}' not found in nodes or edges.")

    def _match_patterns(self):
        combined_results = []
        for pattern in self.patterns:
            source_alias = pattern['source']
            edge_alias = pattern['edge']
            target_alias = pattern['target']
            direction = pattern.get('direction', 'both')  # Default to 'both' if not specified

            source_nodes = self.node_matches[source_alias]['matches']
            target_nodes = self.node_matches[target_alias]['matches']
            edge_matches = self.edge_matches[edge_alias]['matches']

            # Build dictionaries for quick lookup
            source_nodes_dict = {node.getId(): node for node in source_nodes}
            target_nodes_dict = {node.getId(): node for node in target_nodes}

            # Now find matching combinations
            for edge in edge_matches:
                edge_source_id = edge.source
                edge_target_id = edge.target

                # Forward direction: source -> target
                if direction in ('both', 'out'):  # Match outgoing edges from source to target
                    source_node = source_nodes_dict.get(edge_source_id)
                    target_node = target_nodes_dict.get(edge_target_id)
                    if source_node and target_node:
                        result = {
                            source_alias: source_node,
                            edge_alias: edge,
                            target_alias: target_node
                        }
                        combined_results.append(result)

                # Reverse direction: target -> source
                if direction in ('both', 'in'):  # Match incoming edges from target to source
                    source_node = source_nodes_dict.get(edge_target_id)
                    target_node = target_nodes_dict.get(edge_source_id)
                    if source_node and target_node:
                        result = {
                            source_alias: source_node,
                            edge_alias: edge,
                            target_alias: target_node
                        }
                        combined_results.append(result)

        return combined_results
    def _combine_matches(self):
        # Combine matches from nodes and edges without patterns
        combined_results = []

        node_aliases = list(self.node_matches.keys())
        edge_aliases = list(self.edge_matches.keys())

        node_match_lists = [self.node_matches[alias]['matches'] for alias in node_aliases]
        edge_match_lists = [self.edge_matches[alias]['matches'] for alias in edge_aliases]

        for node_combination in product(*node_match_lists):
            result = dict(zip(node_aliases, node_combination))
            if edge_aliases:
                for edge_combination in product(*edge_match_lists):
                    edge_result = result.copy()
                    edge_result.update(dict(zip(edge_aliases, edge_combination)))
                    combined_results.append(edge_result)
            else:
                combined_results.append(result)

        return combined_results

    def _edge_direction_matches(self, edge, node, direction):
        """
        Check if an edge matches the specified direction relative to a given node.

        :param edge: The edge to check.
        :param node: The node relative to which direction is evaluated.
        :param direction: Direction to check ('in', 'out', or 'both').
                """
        if direction == 'both':
            return edge.source == node.getId() or edge.target == node.getId()
        elif direction == 'in':
            return edge.target == node.getId()
        elif direction == 'out':
            return edge.source == node.getId()
        return False

    def _apply_aggregations(self, results):
        if not self.aggregations and not self.groupings:
            return results

        # Group results based on groupings
        grouped_results = defaultdict(list)
        for result in results:
            group_key = tuple(
                grouping(result) if callable(grouping) else result[grouping].getId()
                for grouping in self.groupings
            )
            grouped_results[group_key].append(result)

        # Define aggregation methods for both static values and time series
        agg_funcs = {
            'sum': np.sum,
            'mean': np.mean,
            'min': np.min,
            'max': np.max,
            'count': len  # Include 'count' if needed
        }

        # Process each group
        aggregated_results = []
        for group_key, group_items in grouped_results.items():
            agg_result = {}

            # Include grouping keys in agg_result
            for idx, grouping in enumerate(self.groupings):
                key = group_key[idx]
                if callable(grouping):
                    # Grouping by function (e.g., node property)
                    agg_result['group_key'] = key
                else:
                    # Grouping by alias
                    alias = grouping
                    agg_result[alias] = group_items[0][alias]

            # Apply aggregations for each specified property
            for aggregation in self.aggregations:
                agg_alias = aggregation['alias']
                property_name = aggregation['property_name']
                method = aggregation['method']
                direction = aggregation.get('direction', 'both')
                fill_value = aggregation.get('fill_value', 0)

                # Determine elements to aggregate over
                elements = []
                if agg_alias in self.node_matches:
                    # Aggregating over nodes
                    elements = [item[agg_alias] for item in group_items if agg_alias in item]
                elif agg_alias in self.edge_matches:
                    # Aggregating over edges, consider direction
                    if self.groupings:
                        # Assume the central node is the first grouping alias (if it's an alias)
                        grouping = self.groupings[0]
                        if not callable(grouping):
                            central_node_alias = grouping
                            central_node = group_items[0][central_node_alias]
                            elements = [
                                item[agg_alias]
                                for item in group_items
                                if self._edge_direction_matches(item[agg_alias], central_node, direction)
                            ]
                        else:
                            # Cannot determine central node when grouping by function
                            elements = [item[agg_alias] for item in group_items if agg_alias in item]
                    else:
                        elements = [item[agg_alias] for item in group_items if agg_alias in item]
                else:
                    # Alias not found, skip
                    continue
                    # Handle 'count' method separately
                if method == 'count':
                    agg_result[agg_alias] = len(elements)
                else:

                # Separate handling for time series and static properties
                    if  elements:
                        try:
                            temporal_property = elements[0].get_temporal_property(property_name,0)
                            if temporal_property and isinstance(temporal_property, TimeSeries):
                                # Time series aggregation
                                time_series_list = [
                                    element.get_temporal_property(property_name,0)
                                    for element in elements if element.get_temporal_property(property_name,0)
                                ]
                                if time_series_list:
                                    agg_result[property_name] = TimeSeries.aggregate_multiple(
                                        time_series_list, method
                                    )

                        except (ValueError):
                            # Static properties aggregation
                            values = [
                                element.get_static_property(property_name)
                                for element in elements if element.get_static_property(property_name) is not None
                            ]
                            if values:
                                agg_result[property_name] = agg_funcs[method](values)

            aggregated_results.append(agg_result)

        return aggregated_results

    def _apply_ordering(self, results):
        for ordering in reversed(self.orderings):
            key = ordering['key']
            ascending = ordering['ascending']
            results.sort(key=lambda x: x.get(key, None), reverse=not ascending)
        return results

    def _apply_limit(self, results):
        if self.limit_count is not None:
            results = results[:self.limit_count]
        return results

    def _apply_distinct(self, results):
        if self.distinct_flag:
            unique_results = []
            seen = set()
            for result in results:
                result_tuple = tuple((alias, element.oid) for alias, element in result.items())
                if result_tuple not in seen:
                    seen.add(result_tuple)
                    unique_results.append(result)
            return unique_results
        return results

    def _format_results(self, results):
        formatted_results = []
        for result in results:
            formatted_result = {}
            for alias, func in self.return_elements:
                value = func(result)
                formatted_result[alias] = value
                # Additional debug: Print the computed value for each alias
            formatted_results.append(formatted_result)
        return formatted_results
