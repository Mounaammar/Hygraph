# subgraph_metrics.py
import numpy as np

def count_edges_over_time(hygraph, element_type, oid, attribute, date):
    if element_type == 'subgraph':
        subgraph = hygraph.get_element('subgraph', oid)
        return sum(1 for _, _, edge_data in subgraph.edges(data=True) if edge_data['data'].start_time <= date)
    return 0

def count_nodes_over_time(hygraph, element_type, oid, attribute, date):
    if element_type == 'subgraph':
        subgraph = hygraph.get_element('subgraph', oid)
        return sum(1 for node_data in subgraph.nodes(data=True) if node_data['data'].start_time <= date)
    return 0

# Example aggregation of a time series attribute from nodes and edges in a subgraph
def aggregate_attribute_in_subgraph(hygraph, subgraph_id, element_type, attribute, date, aggregation='sum'):
    values = []

    # Determine whether to process nodes or edges
    if element_type == 'node':
        elements = hygraph.graph.nodes(data=True)
    elif element_type == 'edge':
        elements = hygraph.graph.edges(data=True)
    else:
        raise ValueError(f"Unsupported element_type: {element_type}. Use 'node' or 'edge'.")

    # Iterate over elements (nodes or edges) and collect attribute values
    for element_id, element_data in elements:
        data = element_data['data'] if element_type == 'node' else element_data
        if subgraph_id in data.membership:
            tsid = data.properties.get(attribute)
            if tsid and tsid in hygraph.time_series:
                ts_data = hygraph.time_series[tsid].data
                value = ts_data.sel(time=date, method='ffill').item()
                values.append(value)

    # Perform the desired aggregation
    if aggregation == 'sum':
        return np.sum(values)
    elif aggregation == 'avg':
        return np.mean(values)
    elif aggregation == 'min':
        return np.min(values)
    elif aggregation == 'max':
        return np.max(values)
    elif aggregation == 'count':
        return len(values)
    else:
        raise ValueError(f"Unsupported aggregation: {aggregation}. Supported values are 'sum', 'avg', 'min', 'max', 'count'.")









