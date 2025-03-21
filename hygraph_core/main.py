import cProfile
import pstats
from datetime import datetime, timedelta
import time

import numpy as np
import pandas as pd
from bokeh.server.tornado import psutil
from watchdog.observers import Observer

from HyGraphFileLoaderBatch import HyGraphBatchProcessor
from hygraph import HyGraph, Edge, PGNode, HyGraphQuery
from fileProcessing import NodeFileHandler, EdgeFileHandler, HyGraphFileLoader
import os

from hygraph_core.hygraph_universal_pipeline import HyGraphUniversalPipeline
from hygraph_core.timeseries_operators import TimeSeries, TimeSeriesMetadata
from dask.distributed import Client, progress
from dask import delayed, compute
base_dir = os.path.dirname(os.path.abspath(__file__))


def connection_count_aggregate_function(graph, element_type, oid, attribute, date):
    if element_type == 'node':
        print("number of connections ", len(list(graph.graph.neighbors(oid))))
        return len(list(graph.graph.neighbors(oid)))
    return 0


def count_edges_in_subgraph(subgraph, date):
    return sum(1 for _, _, edge_data in subgraph.view.edges(data=True) if edge_data['data'].start_time <= date)


if __name__ == "__main__":
    hy = HyGraph()
    # Initialize a Dask distributed client (adjust settings as needed)
    '''client = Client()
    print("Dask client initialized:", client)
    print("Dashboard available at:", client.dashboard_link)'''

    # 2) Create a universal pipeline
    pipeline = HyGraphUniversalPipeline(hy)

    # Define the CSV field mappings for nodes & edges:
    # Suppose your node CSV has columns: 'id','start_time','end_time','num_bikes_available',...
    # and your edge CSV has columns: 'id','source_id','target_id','start_time','end_time','num_rides',...
    '''node_field_map = {
        "oid": "station_id",  # CSV "id" -> node's internal ID
        "start_time": "start",
        "end_time": "end"
    }
    edge_field_map = {
        "oid": "id",  # CSV "id" -> edge's internal ID
        "source_id": "from",
        "target_id": "to",
        "start_time": "start",
        "end_time": "end"
    }'''

    # If certain columns are time-series data for each row:
    node_ts_columns = ["num_bikes_available", "num_docks_disabled","num_bikes_disabled","num_bikes_available"]
    edge_ts_columns = ["num_rides", "member_rides","casual_rides","classic_rides","electric_rides","active_trips"]

    # 3) Configure pipeline for JSON
    ''' pipeline.configure_for_json(
        node_json_path=os.path.join(base_dir, 'inputFiles', 'nodes'),
        edge_json_path=os.path.join(base_dir, 'inputFiles', 'edges'),
        node_field_map=node_field_map,
        edge_field_map=edge_field_map
    )
    pipeline.run_pipeline()'''



    # Define the CSV field mappings for nodes & edges:
    # Suppose your node CSV has columns: 'id','start_time','end_time','num_bikes_available',...
    # and your edge CSV has columns: 'id','source_id','target_id','start_time','end_time','num_rides',...
    node_field_map = {
        "oid": "station_id",  # CSV "id" -> node's internal ID
        "start_time": "start_time",
    }
    edge_field_map = {
        "oid": "id",  # CSV "id" -> edge's internal ID
        "source_id": "source_id",
        "target_id": "target_id",
        "start_time": "start_time",
        "end_time": "end_time"
    }

    # 3) Configure pipeline for CSV

    pipeline.configure_for_csv(
        nodes_folder=os.path.join('upload_files', 'data_scale_1_2263_3778433', 'nodes'),
        edges_folder=os.path.join('upload_files', 'data_scale_1_2263_3778433', 'edges'),
        node_field_map=node_field_map,
        edge_field_map=edge_field_map,
        max_rows_per_batch=10_000
    )
    cProfile.run('pipeline.run_pipeline()', 'profile_stats')
    p = pstats.Stats('profile_stats')
    p.sort_stats('cumtime').print_stats(10)
    print("\nFinal HyGraph Summary:")
    print(f"Total nodes loaded: {len(hy.graph.nodes)}")
    print(f"Total edges loaded: {len(hy.graph.edges)}")
    print(f"Total time series objects: {len(hy.time_series)}")

    target_name = "Whitehall St & Bridge St"

    # Using the built-in method get_nodes_by_static_property(property_name, condition):
    matching_nodes = hy.get_nodes_by_static_property(
        property_name="name",
        condition=lambda static_prop: static_prop.value == target_name
    )


    def measure_memory_mb():
        """Return current memory usage in MB using psutil."""
        p = psutil.Process(os.getpid())
        return p.memory_info().rss / (1024 * 1024)


    def timed_run(func, *args, **kwargs):
        """Measure elapsed time and memory usage of func(*args, **kwargs)."""
        m_before = measure_memory_mb()
        t_start = time.time()
        result = func(*args, **kwargs)
        t_end = time.time()
        m_after = measure_memory_mb()
        return (t_end - t_start, m_before, m_after, result)


    def active_ride_aggregator(hy, element_type, oid, attribute, dt):
        """
        Returns 1 if dt is within the ride's [start_time, end_time), else 0.
        """
        edge_obj = hy.get_element(element_type, oid)
        st = getattr(edge_obj, 'start_time', None)
        print()
        et = getattr(edge_obj, 'end_time', None)
        print('st ', st, 'et ',et)
        if st and dt < st:
            return 0
        if et and dt >= et:
            return 0
        return 1


    query_agg = {
        'element_type': 'edge',
        'edge_filter': lambda ed: ed['data'].label == 'trip_concat',
        'aggregation': True,  # merges edges by (source, target, label)
        'edge_settings': {
            'group_by': ['source', 'target', 'label']
        },
        'time_series_config': {
            'start_date': datetime(2023, 1, 1),  # or min from your data
            'end_date': datetime(2025, 2, 1),
            'freq': 'D',
            'attribute': 'active_ride_count',
            'aggregate_function': active_ride_aggregator,
            'use_actual_timestamps': True
        }
    }

    query_noagg = {
        'element_type': 'edge',
        'edge_filter': lambda ed: ed['data'].label == 'trip_concat',
        'aggregation': False,
        'time_series_config': {
            'start_date': datetime(2023, 1, 1),
            'end_date': datetime(2025, 2, 1),
            'freq': 'D',
            'attribute': 'active_ride_count',
            'aggregate_function': active_ride_aggregator,
            'use_actual_timestamps': True
        }
    }
    def run_extract_ts_experiment(iterations=10):
        times_agg = []
        mem_deltas_agg = []
        times_noagg = []
        mem_deltas_noagg = []

        for i in range(iterations):
            print(f"Iteration {i + 1}/{iterations} - Aggregation Query:")
            t_agg, mem_before_agg, mem_after_agg, _ = timed_run(hy.create_time_series_from_graph, query_agg)
            times_agg.append(t_agg)
            mem_deltas_agg.append(mem_after_agg - mem_before_agg)
            print(f"[AGGREGATION] time={t_agg:.2f}s, mem_delta={(mem_after_agg - mem_before_agg):.2f}MB")

            # Optionally, reset or clear the time series from the HyGraph if needed between runs

            print(f"Iteration {i + 1}/{iterations} - No Aggregation Query:")
            t_noagg, mem_before_noagg, mem_after_noagg, _ = timed_run(hy.create_time_series_from_graph, query_noagg)
            times_noagg.append(t_noagg)
            mem_deltas_noagg.append(mem_after_noagg - mem_before_noagg)
            print(f"[NO-AGGREGATION] time={t_noagg:.2f}s, mem_delta={(mem_after_noagg - mem_before_noagg):.2f}MB")

        avg_time_agg = sum(times_agg) / iterations
        avg_mem_agg = sum(mem_deltas_agg) / iterations
        avg_time_noagg = sum(times_noagg) / iterations
        avg_mem_noagg = sum(mem_deltas_noagg) / iterations

        print(f"\nAverage Aggregation: time={avg_time_agg:.2f}s, mem_delta={avg_mem_agg:.2f}MB")
        print(f"Average No Aggregation: time={avg_time_noagg:.2f}s, mem_delta={avg_mem_noagg:.2f}MB")

        return {
            'agg_time': avg_time_agg,
            'agg_mem': avg_mem_agg,
            'noagg_time': avg_time_noagg,
            'noagg_mem': avg_mem_noagg,
        }


    result_data = run_extract_ts_experiment()
    '''nodes_folder = os.path.join(base_dir, 'inputFiles', 'nodes')
    edges_folder = os.path.join(base_dir, 'inputFiles', 'edges')
    subgraph_folder = os.path.join(base_dir, 'inputFiles', 'subgraphs')
    edges_membership_path = base_dir+"/inputFiles/edge_membership.csv"
    nodes_membership_path = base_dir+"/inputFiles/node_membership.csv"

    # Initialize the file loader with the directories and files
    loader = HyGraphBatchProcessor(
        nodes_folder=nodes_folder,
        edges_folder=edges_folder,
        subgraphs_folder=subgraph_folder,
        edges_membership=edges_membership_path,
        nodes_membership=nodes_membership_path
    )

    # Define a condition to filter edges
    def node_filter(node):
        node_obj = node['data']
        print("node filter here ok : ", int(node_obj.get_static_property('age')) > 14)
        return node_obj.label == 'Person' and int(node_obj.get_static_property('age'))  > 14

    def edge_filter(edge):
        edge_obj = edge['data']
        return edge_obj.label == 'Knows'
    # Define the first query for nodes
    query_node = {
        'element_type': 'node',
        'node_filter': node_filter,
        'edge_filter': edge_filter,
        'time_series_config': {
            'start_date': datetime(2023, 1, 1),
            'end_date': datetime(2023, 12, 31),
            'attribute': 'number_connections',
            'aggregate_function': connection_count_aggregate_function,
            'freq': 'M',  # Weekly frequency
            'direction': 'both'  # Direction for edges
        }
    }
    loader.process_batches()  # Process all batches
    # Add the queries to the HyGraph instance
    loader.hygraph.add_query(query_node)
    loader.hygraph.display()
    # Batch processing trigger based on user input
    while True:
        try:
            user_input = input("Type 'run' to start batch processing: ")
            if user_input == 'run':
                print("Starting batch processing...")
                loader.process_batches()  # Process all batches
                loader.hygraph.batch_process()
                loader.hygraph.display()
                print("Batch processing completed.")
        except KeyboardInterrupt:
            print("Batch processing terminated.")
            break'''


#out of files :
    # Sample data for time series
    '''timestamps = ['2024-10-01', '2024-10-02', '2024-10-03']
    variables = ['temperature']
    data = [[25], [27], [28]]

    # Add a time series in the hygraph
    graph_element.add_temporal_property('temperature', timestamps, variables, data)'''

    '''hygraph = HyGraph()  # Initialize an empty HyGraph instance
    #Add mock PGNode stations with static properties including 'capacity'
    node1 = hygraph.add_pgnode(oid=1, label='Station', start_time=datetime.now()- timedelta(hours=7),
                               properties={'capacity': 100, 'name': 'Station A'})
    node2=hygraph.add_pgnode(oid=2, label='Station', start_time=datetime.now()- timedelta(hours=8),
                       properties={'capacity': 40, 'name': 'Station B'})
    node3=hygraph.add_pgnode(oid=3, label='Station', start_time=datetime.now()- timedelta(hours=7),
                       properties={'capacity': 60, 'name': 'Station C'})

    try:
        data = hygraph.graph.nodes[1]
        print("Data keys for node:", data.keys())
        print("Specific data content:", {k: data[k] for k in data.keys()})
    except Exception as e:
        print(f"Error accessing data for node {1}: {e}")



    edge1=hygraph.add_pgedge(oid=4,source=1,target=2,label='Trip', start_time=datetime.now() - timedelta(hours=4))
    edge2 = hygraph.add_pgedge(oid=5, source=1, target=3, label='Trip', start_time=datetime.now() - timedelta(hours=3))
    # Create a TimeSeries object
    timestamps = ['2023-01-01', '2023-01-02', '2023-01-03']
    data = [[10], [20], [15]]  # Values associated with the timestamps
    variables = ['BikeAvailability']

    time_series = hygraph.add_time_series(timestamps, variables, data)

    time_series.display_time_series()
    node1.add_temporal_property("bikeavailable", time_series, hygraph)
    node1.add_static_property("lat", 20, hygraph)
    edge1.add_static_property("bike_type","electric",hygraph)
    print('Node with label station: ', hygraph.get_nodes_by_label('Station'))


    def condition(ts):
        return ts.sum() > 40


    print("here is the first condition ", hygraph.get_nodes_by_temporal_property("bikeavailable", condition))


    def condition_static(static_prop):
        return static_prop.get_value() == 60


    print("here is the second condition ", hygraph.get_nodes_by_static_property("capacity", condition_static))

    query = HyGraphQuery(hygraph)


    def condition_func(node):
        print("Node structure:", node.get('properties', {}).get('capacity').get_value()> 50)  # This will show you what `node` contains
        return node.get('properties', {}).get('capacity').get_value()> 50


    results = (
        query
        .match_node(alias='station', node_id=1)  # Match edge by key only
        .return_(
            name=lambda n: n['station'].get_static_property('name'),
        )
        .execute()
    )
    print("the edge",edge1)
    for result in results:
        print(result)

    print('actua one : ', len(results))
    for node in results:
        print(node)
    # Retrieve the historical in-degree of a node
    ts_in_degree=node_degree_history = hygraph.get_node_degree_over_time(node_id=1, degree_type='in', return_type='history')
    node_degree_history.display_time_series()

    # Retrieve the current out-degree of a node
    current_out_degree = hygraph.get_node_degree_over_time(node_id=1, degree_type='out', return_type='current')
    print(f"Current Out-Degree: {current_out_degree}")
    ts_out_degree=hygraph.get_node_degree_over_time(node_id=1,degree_type='out',return_type='history')

    both_degree= ts_in_degree.aggregate_time_series_cumulative(ts_out_degree,'both_degree').display_time_series()

    hygraph.add_pgedge(oid=60,source=2,target=1,label='Trip',start_time=datetime.now() + timedelta(hours=5))
    ts_in_degree = node_degree_history = hygraph.get_node_degree_over_time(node_id=1, degree_type='in',
                                                                           return_type='history')
    ts_out_degree = hygraph.get_node_degree_over_time(node_id=1, degree_type='out', return_type='history')
    ts_in_degree.aggregate_time_series_cumulative(ts_out_degree,'both_degree').display_time_series()'''

    '''subgraph_original=hygraph.add_subgraph('manhattan', label='Manhattan Subgraph',start_time=datetime(2023, 1, 1))
    subgraph_original['data'].add_static_property("text",230,hygraph)
    t1 = datetime.now() - timedelta(hours=3)
    manhattan_station_ids = [1,2,3]  # IDs of stations in Manhattan at time t1
    manhattan_edge_ids = [4,5]  # IDs of edges (trips) in Manhattan at time t1
    # Add memberships for nodes
    for node_id in manhattan_station_ids:
        hygraph.add_membership(node_id, t1, ['manhattan'], 'node')

    # Add memberships for edges
    for edge_id in manhattan_edge_ids:
        hygraph.add_membership(edge_id, t1, ['manhattan'], 'edge')
    t2 = datetime.now()
    nodes_to_remove = [1]  # IDs of nodes to remove
    edges_to_remove = [4]  # IDs of edges to remove
    for node_id in nodes_to_remove:
        hygraph.remove_membership(node_id, t2, ['manhattan'], 'node')

    for edge_id in edges_to_remove:
        hygraph.remove_membership(edge_id, t2, ['manhattan'], 'edge')
    # At time t1
    subgraph_t1 = hygraph.get_subgraph_at('manhattan', t1)
    # Display subgraph_t1...

    # At time t2
    subgraph_t2 = hygraph.get_subgraph_at('manhattan', t2)

    station_nodes = hygraph.get_nodes_by_label('Station')
    print("\nNodes with label 'Station':", station_nodes)'''



    # Display subgraph_t2...
    '''nodes_data = [
        {
            'id': 'station_1',
            'label': 'Station',
            'start_time': datetime(2023, 1, 1),
            'end_time': None,
            'properties': {
                'name': 'Station 1',
                'capacity': 25,
                'lat': 40.7128,
                'lon': -74.0060
            }
        },
        {
            'id': 'station_2',
            'label': 'Station',
            'start_time': datetime(2023, 1, 1),
            'end_time': None,
            'properties': {
                'name': 'Station 2',
                'capacity': 30,
                'lat': 40.7138,
                'lon': -74.0070
            }},
            {
            'id': 'station_3',
            'label': 'Station',
            'start_time': datetime(2023, 1, 1),
            'end_time': None,
            'properties': {
                'name': 'User 1',
                'age': 30
            }
        }
    ]

    # Add nodes to HyGraph
    for node_data in nodes_data:
        hygraph.add_pgnode(
            oid=node_data['id'],
            label=node_data['label'],
            start_time=node_data['start_time'],
            end_time=node_data['end_time'],
            properties=node_data['properties'])

    # Sample edge data
    edges_data = [
        {
            'id': 'edge_1',
            'source_id': 'station_1',
            'target_id': 'station_2',
            'label': 'Trip',
            'start_time': datetime(2023, 1, 1),
            'end_time': None,
            'properties': {
                'distance': 1.2  # in kilometers
            }
        }
    ]

    # Add edges to HyGraph
    for edge_data in edges_data:
        hygraph.add_pgedge(
            oid=edge_data['id'],
            source=edge_data['source_id'],
            target=edge_data['target_id'],
            label=edge_data['label'],
            start_time=edge_data['start_time'],
            end_time=edge_data['end_time'],
            properties=edge_data['properties']
        )
    # Sample subgraph data
    subgraph_properties = {
        'description': ('Stations in Manhattan', 'static'),
        'creation_date': (datetime.now(), 'static')
    }

    # Add a subgraph to HyGraph
    hygraph.add_subgraph(
        subgraph_id='manhattan',
        label='Manhattan Subgraph',
        properties=subgraph_properties,
        start_time=datetime(2023, 1, 1)
    )
    # Add node memberships to subgraphs
    hygraph.add_membership(
        element_id='station_1',
        timestamp=datetime(2023, 1, 1),
        subgraph_ids=['manhattan'],
        element_type='node'
    )

    hygraph.add_membership(
        element_id='station_2',
        timestamp=datetime(2023, 1, 1),
        subgraph_ids=['manhattan'],
        element_type='node'
    )
    hygraph.add_membership(element_id='station_3', timestamp=datetime(2023, 3, 1),subgraph_ids=['manhattan'],element_type='node')

    timestamp = datetime(2023, 1, 1)  # Example timestamp

    # Get the subgraph at the given timestamp
    subgraph_at_time = hygraph.get_subgraph_at('manhattan', timestamp)

    # Display the subgraph
    import matplotlib.pyplot as plt
    import networkx as nx
    # Check nodes and edges in the subgraph
    print(f"Nodes in subgraph at {timestamp}: {list(subgraph_at_time.nodes())}")
    print(f"Edges in subgraph at {timestamp}: {list(subgraph_at_time.edges())}")

    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(subgraph_at_time, k=0.15, iterations=20)
    nx.draw_networkx_nodes(subgraph_at_time, pos, node_size=50, node_color='blue')
    nx.draw_networkx_edges(subgraph_at_time, pos, alpha=0.3)
    nx.draw_networkx_labels(subgraph_at_time, pos, font_size=8)
    plt.title(f"Subgraph 'manhattan' at {timestamp}")
    plt.axis('off')
    plt.show()'''


    '''   def create_trip_series(start_time, length=10):
        timestamps = [start_time + timedelta(minutes=5 * i) for i in range(length)]
        trip_counts = np.random.randint(5, 20, size=length)
        data = trip_counts.reshape((length, 1))  # Reshape to (length, 1)
        metadata = TimeSeriesMetadata(owner_id=None)

        return hygraph.add_time_series(
            timestamps=timestamps,
            variables=['trip_count'],
            data=data,
            metadata=metadata
        )


    fixed_date = datetime(2023, 1, 1, 0, 0, 0)
    ts3= create_trip_series(start_time=fixed_date)
    ts4 = create_trip_series(start_time=fixed_date)

    # Create sample time series data
    timestamps1 = pd.date_range(start='2023-01-01', periods=5, freq='D')
    timestamps2 = pd.date_range(start='2023-01-01', periods=5, freq='D')
    variables = ['var1', 'var2']
    data1 = np.random.rand(5, 2)
    data2 = np.random.rand(5, 2)

    metadata= TimeSeriesMetadata(1)
    # Create TimeSeries instances
    ts1 = TimeSeries(tsid='ts1', timestamps=timestamps1, variables=variables, data=data1,metadata=metadata)
    ts2 = TimeSeries(tsid='ts2', timestamps=timestamps2, variables=variables, data=data2,metadata=metadata)
    ts3.display_time_series()
    ts4.display_time_series()
    # Compute similarity measures
    print("Euclidean Distance:", ts3.euclidean_distance(ts4))
    print("Correlation Coefficient:", ts4.correlation_coefficient(ts3))
    print("Cosine Similarity:", ts3.cosine_similarity(ts4))
    print("DTW Distance:", ts4.dynamic_time_warping(ts3,'trip_count'))
    print('DTW MUltivariate', ts1.dtw_independent_multivariate(ts1))'''

