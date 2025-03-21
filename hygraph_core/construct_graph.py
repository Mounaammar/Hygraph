from datetime import datetime

import networkx as nx
import numpy as np
import pandas as pd
from fastdtw import fastdtw
from scipy.stats import pearsonr

from hygraph_core.graph_operators import TSNode
from hygraph_core.hygraph import HyGraph, HyGraphQuery
from hygraph_core.timeseries_operators import TimeSeriesMetadata, TimeSeries


def build_timeseries_similarity_graph(time_series_list, threshold, node_label, ts_attr_list, variable_name, hygraph=None,
                                     shape_similarity_metric='euclidean', feature_similarity_metric='cosine',
                                      similarity_weights=None, edge_type='PGEdge',
                                     ):
    """
    Build a HyGraph where each time series is a TSNode, and edges are added based on combined similarity.

    :param time_series_list: List of dictionaries containing time series data. Each dict should have:
        - 'timestamps': List of timestamps
        - 'variables': List of variable names (for multivariate time series)
        - 'data': 2D list or numpy array of data corresponding to variables
    :param threshold: Similarity threshold to add an edge between two nodes
    :param node_label: Label for the nodes
    :param ts_attr_list: List of attributes for the time series nodes
    :param hygraph: An existing HyGraph instance or None to create a new one
    :param shape_similarity_metric: Metric for shape similarity ('euclidean', 'dtw', 'correlation', etc.)
    :param feature_similarity_metric: Metric for feature similarity ('cosine', 'euclidean', etc.)
    :param similarity_weights: Dictionary with weights for 'shape' and 'feature' similarities
    :param edge_type: Type of edge to create ('PGEdge' or 'TSEdge')
    :param variable_name: Variable name to use for DTW if time series are multivariate
    :return: A HyGraph instance with TSNodes and edges based on similarity
    """
    if similarity_weights is None:
        similarity_weights = {'shape': 0.5, 'feature': 0.5}
    if hygraph is None:
        hygraph = HyGraph()

    ts_nodes = []
    tsid_to_nodeid = {}

    # Step 1: Create TSNodes for each time series
    for idx, ts_data in enumerate(time_series_list):
        tsid = hygraph.id_generator.generate_timeseries_id()
        node_id = hygraph.id_generator.generate_node_id()

        # Create TimeSeries object
        timestamps = ts_data['timestamps']
        variables = ts_data['variables']
        data = ts_data['data']
        metadata = TimeSeriesMetadata(owner_id=node_id, element_type='TSNode', attributes=ts_attr_list[idx])

        time_series = TimeSeries(tsid=tsid, timestamps=timestamps, variables=variables, data=data, metadata=metadata)
        hygraph.time_series[tsid] = time_series

        # Create TSNode
        ts_node = hygraph.add_tsnode(oid=node_id, label=node_label, time_series=time_series)

        ts_nodes.append(ts_node)
        tsid_to_nodeid[tsid] = node_id

    # Step 2: Compute pairwise similarities and add edges
    num_ts = len(ts_nodes)
    for i in range(num_ts):
        for j in range(i + 1, num_ts):
            ts1 = ts_nodes[i].series  # TimeSeries object
            ts2 = ts_nodes[j].series  # TimeSeries object

            # Compute shape similarity using existing TimeSeries methods
            shape_similarity = ts1.shape_similarity(ts2, variable_name=variable_name, metric=shape_similarity_metric)

            # Compute feature similarity using existing TimeSeries methods
            feature_similarity = ts1.feature_similarity(ts2, metric=feature_similarity_metric)

            # Combine similarities using specified weights
            total_similarity = (similarity_weights['shape'] * shape_similarity) + \
                               (similarity_weights['feature'] * feature_similarity)

            if total_similarity >= threshold:
                # Determine overlapping time period for the edge
                edge_start_time = max(ts1.first_timestamp(), ts2.first_timestamp())
                edge_end_time = min(ts1.last_timestamp(), ts2.last_timestamp())
                # Create an edge between ts_nodes[i] and ts_nodes[j]
                edge_id = hygraph.id_generator.generate_edge_id()

                if edge_type == 'PGEdge':
                    # For PGEdge, store the similarities as properties
                    properties = {
                        'total_similarity': total_similarity,
                        'shape_similarity': shape_similarity,
                        'feature_similarity': feature_similarity
                    }
                    hygraph.add_pgedge(edge_id, ts_nodes[i].getId(), ts_nodes[j].getId(), 'Similar',
                                       start_time=edge_start_time,end_time=edge_end_time, properties=properties)
                elif edge_type == 'TSEdge':
                    # For TSEdge, store the similarity over time as a TimeSeries
                    similarity_ts = compute_similarity_timeseries(ts1, ts2, metric=shape_similarity_metric)
                    hygraph.add_tsedge(edge_id, ts_nodes[i].getId(), ts_nodes[j].getId(),
                                       'Similar', similarity_ts)
                else:
                    raise ValueError("edge_type must be 'PGEdge' or 'TSEdge'")

    return hygraph


# Utility functions
def compute_similarity_timeseries(ts1, ts2, metric='euclidean', window_size=5):
    """
    Compute similarity over time between two TimeSeries objects using a sliding window.

    :param ts1: First TimeSeries object
    :param ts2: Second TimeSeries object
    :param metric: Similarity metric ('euclidean', 'dtw', 'correlation')
    :param window_size: Size of the sliding window
    :return: TimeSeries object representing similarity over time
    """
    timestamps = []
    similarities = []

    data1 = ts1.data.values
    data2 = ts2.data.values
    times1 = ts1.data.coords['time'].values
    times2 = ts2.data.coords['time'].values

    # Ensure same timestamps
    common_times = np.intersect1d(times1, times2)
    idx1 = np.isin(times1, common_times)
    idx2 = np.isin(times2, common_times)
    data1 = data1[idx1]
    data2 = data2[idx2]

    for i in range(len(common_times) - window_size + 1):
        window_data1 = data1[i:i + window_size]
        window_data2 = data2[i:i + window_size]
        timestamp = common_times[i + window_size - 1]

        # Compute similarity in the window
        ts_window1 = TimeSeries(tsid=None, timestamps=common_times[i:i + window_size],
                                variables=ts1.data.coords['variable'].values,
                                data=window_data1)
        ts_window2 = TimeSeries(tsid=None, timestamps=common_times[i:i + window_size],
                                variables=ts2.data.coords['variable'].values,
                                data=window_data2)
        similarity = ts_window1.shape_similarity(ts_window2, metric=metric)

        timestamps.append(timestamp)
        similarities.append([similarity])

    variables = ['similarity']
    tsid = None  # Since this is a temporary TimeSeries, we can set tsid to None
    metadata = TimeSeriesMetadata(owner_id=None, element_type='TSEdge')
    similarity_ts = TimeSeries(tsid=tsid, timestamps=timestamps,
                               variables=variables, data=similarities, metadata=metadata)

    return similarity_ts

if __name__ == '__main__':
    # Sample time series data for stocks
    # Generate sample time series data for 10 stocks
    '''time_series_list = []
    node_label_list = []

    # Common timestamps
    timestamps = pd.date_range(start='2023-01-01', periods=100, freq='D')

    # Base patterns
    base_pattern = np.sin(np.linspace(0, 2 * np.pi, 100))

    # Create 10 stocks with variations
    for i in range(10):
        # Slightly modify the base pattern for each stock
        noise = np.random.normal(0, 0.1, 100)
        shift = np.random.uniform(-0.5, 0.5)
        data = (base_pattern + shift + noise).reshape(-1, 1)
        ts_data = {
            'timestamps': timestamps,
            'variables': ['Price'],
            'data': data
        }
        time_series_list.append(ts_data)
        node_label_list.append(f"Stock_{chr(65 + i)}")  # Labels: Stock_A, Stock_B, ..., Stock_J

    # Build the HyGraph
    threshold = 0.95  # High similarity threshold
    hygraph = build_timeseries_similarity_graph(
        time_series_list,
        threshold=threshold,
        node_label='Stock',
        ts_attr_list=node_label_list,
        similarity_metric='correlation',  # Using correlation as the similarity metric
        edge_type='PGEdge'
    )
    # Print the nodes and edges
    # Print the nodes
    print("Nodes:")
    for node_id, data in hygraph.graph.nodes(data=True):
        node = data.get('data')
        ts=hygraph.get_timeseries(node.series.tsid)
        print(f"Node ID: {node_id}, Label: {node.label}, TSID: {node.series.tsid}, Attribute: {ts.metadata.attribute}")

    # Print the edges
    print("\nEdges:")
    for u, v, key, data in hygraph.graph.edges(keys=True, data=True):
        edge = data.get('data')
        similarity = edge.get_static_property('similarity')
        print(f"Edge ID: {edge.oid}, From: {u}, To: {v}, Similarity: {similarity:.4f}")

    # Apply community detection (e.g., Girvan-Newman algorithm)
    communities = nx.algorithms.community.girvan_newman(hygraph.graph)
    # Get the first level of communities
    top_level_communities = next(communities)
    community_list = [list(c) for c in top_level_communities]

    # Print the clusters
    for idx, community in enumerate(community_list):
        stock_names = [hygraph.graph.nodes[node]['data'].label for node in community]
        print(f"Community {idx + 1}: {stock_names}")

    # Calculate centrality measures
    centrality = nx.degree_centrality(hygraph.graph)

    # Get the top N central stocks
    top_central_stocks = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]

    # Print the most central stocks
    for node_id, centrality_value in top_central_stocks:
        stock = hygraph.graph.nodes[node_id]['data']
        print(f"Stock: {stock.label}, Centrality: {centrality_value:.4f}")

    # Retrieve the historical in-degree of a node
    query = HyGraphQuery(hygraph)

    stockA = hygraph.get_nodes_by_label('')
    ts_in_degree = node_degree_history = hygraph.get_node_degree_over_time(node_id=1, degree_type='in',
                                                                           return_type='history')
    node_degree_history.display_time_series()'''

    # Sample time series data for stocks
    time_series_list = []
    ts_attr_list = []

    # Common timestamps
    timestamps = pd.date_range(start='2023-01-01', periods=100, freq='D')

    # Base patterns
    base_pattern = np.sin(np.linspace(0, 2 * np.pi, 100))

    # Define companies and domains
    companies = [
        {'name': 'Company_A', 'domain': 'Tech'},
        {'name': 'Company_B', 'domain': 'Finance'},
        {'name': 'Company_C', 'domain': 'Healthcare'},
        {'name': 'Company_D', 'domain': 'Tech'},
        {'name': 'Company_E', 'domain': 'Finance'},
        {'name': 'Company_F', 'domain': 'Healthcare'},
        {'name': 'Company_G', 'domain': 'Tech'},
        {'name': 'Company_H', 'domain': 'Finance'},
        {'name': 'Company_I', 'domain': 'Healthcare'},
        {'name': 'Company_J', 'domain': 'Tech'}
    ]

    # Create 10 stocks with variations
    for i in range(len(companies)):
        # Slightly modify the base pattern for each stock
        noise = np.random.normal(0, 0.1, 100)
        shift = np.random.uniform(-0.5, 0.5)
        data = (base_pattern + shift + noise).reshape(-1, 1)
        ts_data = {
            'timestamps': timestamps,
            'variables': ['Price'],
            'data': data
        }
        time_series_list.append(ts_data)

        # Create attributes dictionary for TimeSeriesMetadata
        attributes = {
            'company_name': companies[i]['name'],
            'domain': companies[i]['domain']
        }
        ts_attr_list.append(attributes)

    # Build the HyGraph
    threshold = 0.96 # Similarity threshold

    hygraph = build_timeseries_similarity_graph(
        time_series_list=time_series_list,
        threshold=threshold,
        node_label='Stock',
        ts_attr_list=ts_attr_list,
        variable_name='Price',  # Assuming 'Price' is the variable name
        shape_similarity_metric='euclidean',
        feature_similarity_metric='cosine',
        similarity_weights=None,
        edge_type='PGEdge'
    )
    # Print the nodes and edges
    # Print the nodes
    print("Nodes:")
    for node_id, data in hygraph.graph.nodes(data=True):
        node = data.get('data')
        ts = hygraph.get_timeseries(node.series.tsid)
        print(f"Node ID: {node_id}, Label: {node.label}, TSID: {node.series.tsid}, Attribute: {ts.metadata.attributes}")

    # Print the edges
    print("\nEdges:")
    for u, v, key, data in hygraph.graph.edges(keys=True, data=True):
        edge = data.get('data')
        similarity = edge.get_static_property('total_similarity')
        print(f"Edge ID: {edge.oid}, From: {u}, To: {v}, Similarity: {similarity:.4f}")

    # Create subgraphs for each domain
    # Get the list of domains
    domains = set(company['domain'] for company in companies)
    timestamp = pd.to_datetime('2023-01-01')
    # Create a subgraph for each domain
    for domain in domains:
        # Filter nodes that belong to the current domain
        def node_filter(node_id, data):
            node = data.get('data')
            if node and node.label == 'Stock':
                ts = hygraph.get_timeseries(node.series.tsid)
                attributes = ts.metadata.attributes
                return attributes.get('domain') == domain
            return False


        def edge_filter(u, v, key, data):
            node_u_data = hygraph.graph.nodes[u]['data']
            node_v_data = hygraph.graph.nodes[v]['data']
            if node_u_data and node_v_data:
                domain_u = node_u_data.series.metadata.attributes.get('domain')
                domain_v = node_v_data.series.metadata.attributes.get('domain')
                return domain_u == domain or domain_v == domain # Both nodes are in the same domain
            return False


        subgraph_id = f"Subgraph_{domain}"
        hygraph.add_subgraph(
            subgraph_id=subgraph_id,
            label=domain,
            node_filter=node_filter,
            edge_filter=edge_filter,
            start_time=timestamp
        )


    hygraph.display_subgraph('Subgraph_Tech', timestamp)
    # Function to get node IDs in a subgraph
    def get_subgraph_node_ids(subgraph_id):
        subgraph = hygraph.get_subgraph_at(subgraph_id, datetime.now())
        return set(subgraph.nodes())


    source_domain = 'Tech'  # Replace with your desired domain

    # Initialize the query
    query = HyGraphQuery(hygraph)
    results = (
        query
        .match_node(alias='source_node', label='Stock')
        .where(lambda source_node: source_node.series.metadata.attributes['domain'] == source_domain)
        .match_edge(alias='edge')
        .match_node(alias='target_node', label='Stock')
        .connect('source_node', 'edge', 'target_node')
        .where(lambda target_node: target_node.series.metadata.attributes['domain'] != source_domain)
        .group_by(lambda result: result['target_node'].series.metadata.attributes['domain'] )
        .aggregate(
            alias='edge',
            property_name='count',
            method='count',
            direction='both'  # Adjust direction as needed
        )
        .order_by('edge', ascending=False)
        .limit(1)
        .return_(
            domain=lambda n: n['group_key'],
            count=lambda n: n['edge']
        )
        .execute()
    )

    if results:
        result = results[0]
        most_connected_domain = result['domain']
        max_count = result['count']
        print(
            f"The domain most connected to '{source_domain}' is '{most_connected_domain}' with {max_count} connections.")
    else:
        print(f"No connections from domain '{source_domain}' to other domains were found.")

    # Step 5: Perform community detection within each domain-specific subgraph
    for domain in domains:
        subgraph_id = f"Subgraph_{domain}"
        subgraph_view = hygraph.subgraphs[subgraph_id]['view']
        # Convert subgraph_view to an undirected graph for community detection
        undirected_subgraph = nx.Graph(subgraph_view)
        # Apply community detection algorithm (e.g., Girvan-Newman)
        communities_gen = nx.algorithms.community.girvan_newman(undirected_subgraph)
        # Get the first level of communities
        top_level_communities = next(communities_gen)
        community_list = [list(c) for c in top_level_communities]

        print(f"\nCommunities in {domain} Domain:")
        for idx, community in enumerate(community_list):
            stock_names = [hygraph.graph.nodes[node]['data'].label for node in community]
            print(f"Community {idx + 1}: {stock_names}")

        # Step 6: Centrality analysis within the subgraph
        centrality = nx.degree_centrality(undirected_subgraph)

        # Get the top N central stocks in this domain
        top_central_stocks = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:3]

        print(f"\nTop Central Stocks in {domain} Domain:")
        for node_id, centrality_value in top_central_stocks:
            stock = hygraph.graph.nodes[node_id]['data']
            print(f"Stock: {stock.label}, Centrality: {centrality_value:.4f}")

    # Additional analysis or community detection can be added here

