"""
hygraph_json_loader.py

An advanced ETL pipeline for loading large JSON *directories* into HyGraph,
using ijson for streaming and user-defined field mappings.

We assume each file in node_json_path is a top-level JSON array of node records,
and each file in edge_json_path is a top-level JSON array of edge records.
The filename (without .json) is used as the "label" for those nodes or edges.

Example:
  node_json_path/
    station.json       # => label="station"
    special_nodes.json # => label="special_nodes"
  edge_json_path/
    super_edge.json    # => label="super_edge"
    ...
"""
import polars as pl

NODES_SCHEMA = {
    "id": pl.Utf8,
    "start_time": pl.Utf8,
    "end_time": pl.Utf8,
}

EDGES_SCHEMA = {
    "id": pl.Utf8,
    "source_id": pl.Utf8,
    "target_id": pl.Utf8,
    "start_time": pl.Utf8,
    "end_time": pl.Utf8,
}
import os
import ijson
import xxhash
from datetime import datetime
from typing import Optional, Dict, Any

from hygraph_core.hygraph import HyGraph
from hygraph_core.timeseries_operators import TimeSeries, TimeSeriesMetadata
# If you have parse_datetime in hygraph_core.constraints, you can use that.
# Otherwise, we define a simple version below.

FAR_FUTURE_DATE = datetime(2100, 12, 31, 23, 59, 59)

def simple_parse_date(date_str: str) -> Optional[datetime]:
    """
    A minimal date parser for ISO-like strings (e.g. "2024-05-16T00:00:00").
    Adjust if your JSON uses a different format.
    """
    if not date_str:
        return None
    try:
        return datetime.fromisoformat(date_str)
    except ValueError:
        return None

def fast_hash(key):
    return xxhash.xxh64(str(key)).intdigest()
class HyGraphJSONLoader:
    """
    A specialized pipeline that can read *directories* containing multiple JSON files
    (one JSON file per label, or multiple) for nodes and edges, and create or update
    HyGraph with time-series logic, user-defined field mappings, etc.
    """

    def __init__(
        self,
        hygraph: HyGraph,
        node_json_path: str,
        edge_json_path: str,
        node_field_map: Dict[str, str],
        edge_field_map: Dict[str, str],

    ):
        """
        :param hygraph: An instance of HyGraph where data is loaded.
        :param node_json_path: Directory of JSON files for node data.
        :param edge_json_path: Directory of JSON files for edge data.
        :param node_field_map: A dict mapping loader fields ('oid','start_time',...)
                               to JSON keys (e.g. "station_id","start",...).
        :param edge_field_map: Similarly for edges.
        """
        self.hygraph = hygraph
        self.node_json_path = node_json_path
        self.edge_json_path = edge_json_path
        self.node_field_map = node_field_map
        self.edge_field_map = edge_field_map
        self.node_hash = {}

    def run_pipeline(self):
        """
        Main pipeline method:
          1. Load all nodes (from the directory)
          2. Load all edges (from the directory)
          3. Print final status
        """
        print("========== Starting JSON ETL Pipeline (directory-based) ==========")
        self.load_nodes()
        self.load_edges()
        self.finalize_pipeline()
        print("========== JSON ETL Pipeline Complete ==========")

    def finalize_pipeline(self):
        """
        Post-processing or final display.
        You might prefer to skip hygraph.display() if the graph is huge,
        and instead query a single node or partial info.
        """
        print("\nFinal HyGraph state after JSON pipeline (partial info).")

        # EXAMPLE: Instead of hygraph.display(), you might just show # of nodes/edges
        node_count = len(self.hygraph.graph.nodes)
        edge_count = len(self.hygraph.graph.edges)
        print(f"Nodes loaded: {node_count}")
        print(f"Edges loaded: {edge_count}")

        # If you want more detail, you can do a partial query, e.g.:
        # matching_nodes = self.hygraph.get_nodes_by_static_property("name",
        #    lambda sp: sp.value == "Whitehall St & Bridge St")
        # ...

    ########################################
    #             LOAD NODES              #
    ########################################

    def load_nodes(self):
        """
        Loads *all* JSON files from the node_json_path directory, each file containing
        a top-level JSON array of node objects.
        The filename (minus .json) is used as the 'label' for these nodes.
        """
        if not os.path.isdir(self.node_json_path):
            print(f"[JSON Loader] Node directory not found: {self.node_json_path}")
            return

        node_files = [
            f for f in os.listdir(self.node_json_path)
            if f.endswith(".json")
        ]
        for file_name in node_files:
            file_label = file_name.replace(".json","")
            file_path = os.path.join(self.node_json_path, file_name)
            print(f"\n[JSON Loader] Loading node file '{file_path}' with label='{file_label}'")
            self._load_node_file(file_path, file_label)


    def _load_node_file(self, file_path: str, label: str):
        node_count = 0
        try:
            with open(file_path, "rb") as f:
                # ijson.items(...) with prefix="item" means we iterate over each array element
                for node_obj in ijson.items(f, "item"):
                    node_count += 1
                    self._process_node_record(node_obj, label)
            print(f"  -> {node_count} nodes processed from '{file_path}'.")
        except Exception as e:
            print(f"[ERROR] Failed to load nodes from {file_path}: {e}")

    def _process_node_record(self, node_obj: Dict[str, Any], label: str):
        oid = normalize_oid(node_obj.get(self.node_field_map.get("oid", ""), f"node_{id(node_obj)}"))
        start_col = self.node_field_map.get("start_time", "start_time")
        end_col = self.node_field_map.get("end_time", "end_time")
        hashed_id = fast_hash(oid)
        self.node_hash[oid] = hashed_id

        # parse start_time and end_time
        start = simple_parse_date(node_obj.get(self.node_field_map.get("start_time", ""), "")) or datetime.now()
        end = simple_parse_date(node_obj.get(self.node_field_map.get("end_time", ""), "")) or FAR_FUTURE_DATE

        #static properties
        known_mapped_keys = {oid, start_col, end_col} # e.g. station_id, start, end, labels, ts
        node_properties = {k: v for k, v in node_obj.items() if k not in known_mapped_keys and k != "ts"}

        # create or update node
        existing_node = None
        if hashed_id in self.hygraph.graph.nodes:
            existing_node = self.hygraph.graph.nodes[hashed_id]["data"]

        if not existing_node:
            self.hygraph.add_pgnode(
                oid=hashed_id,
                label=label,
                start_time=start,
                end_time=end,
                properties=node_properties
            )
        else:
            for kk, vv in node_properties.items():
                existing_node.add_static_property(kk, vv, self.hygraph)

        # Time series logic: if there's a "ts" object, parse & attach as temporal properties
        ts_obj = node_obj.get(self.node_field_map.get("time_series_key", "ts"), {})
        if isinstance(ts_obj, dict):
            self._attach_time_series(hashed_id, ts_obj, "node")


    ########################################
    #             LOAD EDGES              #
    ########################################

    def load_edges(self):
        """
        Loads *all* JSON files from the edge_json_path directory, each file containing
        a top-level JSON array of edge objects. The filename (minus .json) is used as
        the label for these edges.
        """
        if not os.path.isdir(self.edge_json_path):
            print(f"[JSON Loader] Edge directory not found: {self.edge_json_path}")
            return

        edge_files = [
            f for f in os.listdir(self.edge_json_path)
            if f.endswith(".json")
        ]
        for file_name in edge_files:
            file_label = file_name.replace(".json","")
            file_path = os.path.join(self.edge_json_path, file_name)
            print(f"\n[JSON Loader] Loading edge file '{file_path}' with label='{file_label}'")
            self._load_edge_file(file_path, file_label)

    def _load_edge_file(self, file_path: str, label: str):
        edge_count = 0
        try:
            with open(file_path, "rb") as f:
                for edge_obj in ijson.items(f, "item"):
                    edge_count += 1
                    self._process_edge_record(edge_obj, label)
            print(f"  -> {edge_count} edges processed from '{file_path}'.")
        except Exception as e:
            print(f"[ERROR] Failed to load edges from {file_path}: {e}")

    def _process_edge_record(self, edge_obj: Dict[str, Any], label: str):
        # parse or generate edge ID
        oid = normalize_oid(edge_obj.get(self.edge_field_map.get("oid", ""), ""))
        s_val = normalize_oid(edge_obj.get(self.edge_field_map.get("source_id","from"),""))
        t_val = normalize_oid(edge_obj.get(self.edge_field_map.get("target_id","to"),""))
        sid = self.node_hash.get(s_val)
        tid = self.node_hash.get(t_val)
        if not sid or not tid:
            return

        # parse times
        hashed_eid = fast_hash(oid)
        start = simple_parse_date(edge_obj.get(self.edge_field_map.get("start_time", ""), "")) or datetime.now()
        end = simple_parse_date(edge_obj.get(self.edge_field_map.get("end_time", ""), "")) or FAR_FUTURE_DATE

        #static properties
        edge_properties = {k: v for k, v in edge_obj.items() if k not in self.edge_field_map.values() and k != "ts"}

        # ensure source/target exist
        if sid not in self.hygraph.graph.nodes:
            print(f"   [WARN] Edge {hashed_eid}: source node {sid} not found.")
            return
        if tid not in self.hygraph.graph.nodes:
            print(f"   [WARN] Edge {hashed_eid}: target node {tid} not found.")
            return

        # create or update
        existing_edge = None
        if self.hygraph.graph.has_edge(sid, tid, key=hashed_eid):
            existing_edge = self.hygraph.graph[sid][tid][hashed_eid]["data"]

        if not existing_edge:

            self.hygraph.add_pgedge(
                oid=hashed_eid,
                source=sid,
                target=tid,
                label=label,
                start_time=start,
                end_time=end,
                properties=edge_properties
            )
        else:

            for kk, vv in edge_properties.items():
                existing_edge.add_static_property(kk, vv, self.hygraph)

        # handle time-series if "ts" or other key is present
        ts_obj = edge_obj.get(self.edge_field_map.get("time_series_key", "ts"), {})
        if isinstance(ts_obj, dict):
            self._attach_time_series(hashed_eid, ts_obj, "edge")

    def _attach_time_series(self, owner_id: int, ts_obj: Dict[str, Any], element_type: str):

        for ts_name, arr in ts_obj.items():
            if not isinstance(arr, list):
                continue

            tsid = f"{owner_id}_{ts_name}"
            existing_ts = self.hygraph.time_series.get(tsid)
            if not existing_ts:
                # build new
                timestamps = []
                values = []
                for rec in arr:
                    timestamps.append(simple_parse_date(rec.get("Start", "")) or datetime.now())
                    values.append([rec.get("Value", 0)])
                metadata = TimeSeriesMetadata(owner_id=owner_id, element_type=element_type)
                new_ts = TimeSeries(tsid, timestamps, [ts_name], values, metadata)
                self.hygraph.time_series[tsid] = new_ts
                if element_type == "node":
                    self.hygraph.graph.nodes[owner_id]["data"].add_temporal_property(ts_name, new_ts, self.hygraph)
                else:
                    for u, v, k, d in self.hygraph.graph.edges(keys=True, data=True):
                        if k == owner_id:
                            d["data"].add_temporal_property(ts_name, new_ts, self.hygraph)
                            break



def normalize_oid(raw_id):
    try:
        as_float = float(raw_id)
        if as_float.is_integer():
            return str(int(as_float))
        return str(raw_id)
    except (ValueError, TypeError):
        return str(raw_id)
