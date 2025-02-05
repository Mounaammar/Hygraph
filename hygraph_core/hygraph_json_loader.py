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

import os
import ijson
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
        """
        Similar logic as your original approach, but we pass 'label' from the filename
        so you can store that as the node's label if desired.
        """
        external_id = None
        if self.node_field_map.get("oid"):
            key = self.node_field_map["oid"]
            external_id = str(node_obj.get(key, ""))
        if not external_id:
            external_id = f"node_{id(node_obj)}"  # fallback

        # parse start_time
        start = None
        if self.node_field_map.get("start_time"):
            start_str = node_obj.get(self.node_field_map["start_time"], "")
            start = simple_parse_date(start_str) or datetime.now()
        else:
            start = datetime.now()

        # parse end_time
        end = None
        if self.node_field_map.get("end_time"):
            end_str = node_obj.get(self.node_field_map["end_time"], "")
            end = simple_parse_date(end_str) or FAR_FUTURE_DATE
        else:
            end = FAR_FUTURE_DATE

        # parse labels if you want (could be a list in JSON)
        main_label = label  # By default we use the filename's label
        if self.node_field_map.get("labels"):
            labels_json_key = self.node_field_map["labels"]
            maybe_label = node_obj.get(labels_json_key, None)
            if isinstance(maybe_label, str):
                main_label = maybe_label
            elif isinstance(maybe_label, list) and maybe_label:
                main_label = maybe_label[0]

        # leftover static properties
        known_mapped_keys = set(self.node_field_map.values())  # e.g. station_id, start, end, labels, ts
        node_properties = {}
        for k, v in node_obj.items():
            if k not in known_mapped_keys and k not in ("ts",):
                node_properties[k] = v

        # create or update node
        existing_node = None
        if external_id in self.hygraph.graph.nodes:
            existing_node = self.hygraph.graph.nodes[external_id]["data"]

        if not existing_node:
            self.hygraph.add_pgnode(
                oid=external_id,
                label=main_label,
                start_time=start,
                end_time=end,
                properties=node_properties
            )
        else:
            for kk, vv in node_properties.items():
                existing_node.add_static_property(kk, vv, self.hygraph)

        # Time series logic: if there's a "ts" object, parse & attach as temporal properties
        ts_obj_key = self.node_field_map.get("time_series_key", "ts")
        ts_obj = node_obj.get(ts_obj_key, {})
        if isinstance(ts_obj, dict):
            self._process_node_time_series(external_id, ts_obj)

    def _process_node_time_series(self, external_id: str, ts_obj: Dict[str, Any]):
        node_data = self.hygraph.graph.nodes[external_id]["data"]
        for ts_name, arr in ts_obj.items():
            if not isinstance(arr, list):
                continue

            tsid = f"{external_id}_{ts_name}"
            existing_ts = self.hygraph.time_series.get(tsid)
            if not existing_ts:
                # build new
                timestamps = []
                values = []
                for rec in arr:
                    start_str = rec.get("Start", "")
                    val = rec.get("Value", 0)
                    parsed_start = simple_parse_date(start_str) or datetime.now()
                    timestamps.append(parsed_start)
                    values.append([val])
                metadata = TimeSeriesMetadata(owner_id=external_id, element_type="node")
                new_ts = TimeSeries(tsid, timestamps, [ts_name], values, metadata)
                self.hygraph.time_series[tsid] = new_ts
                node_data.add_temporal_property(ts_name, new_ts, self.hygraph)
            else:
                # possibly update or append
                pass

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
        external_id = None
        if self.edge_field_map.get("oid"):
            key = self.edge_field_map["oid"]
            external_id = str(edge_obj.get(key, ""))

        if not external_id:
            # fallback
            s_val = str(edge_obj.get(self.edge_field_map.get("source_id","from"),""))
            t_val = str(edge_obj.get(self.edge_field_map.get("target_id","to"),""))
            st_key = self.edge_field_map.get("start_time", "start")
            st_str = edge_obj.get(st_key, "")
            external_id = f"edge_{s_val}_{t_val}_{st_str}"

        # parse times
        start = None
        if self.edge_field_map.get("start_time"):
            start_str = edge_obj.get(self.edge_field_map["start_time"], "")
            start = simple_parse_date(start_str) or datetime.now()
        else:
            start = datetime.now()

        end = None
        if self.edge_field_map.get("end_time"):
            end_str = edge_obj.get(self.edge_field_map["end_time"], "")
            end = simple_parse_date(end_str) or FAR_FUTURE_DATE
        else:
            end = FAR_FUTURE_DATE

        # parse source & target
        source_key = self.edge_field_map.get("source_id","from")
        target_key = self.edge_field_map.get("target_id","to")
        source_id = str(edge_obj.get(source_key, ""))
        target_id = str(edge_obj.get(target_key, ""))

        # parse label from JSON field if specified, else fallback to the filename label
        label_key = self.edge_field_map.get("label")
        final_label = label  # default is the filename label
        if label_key:
            maybe_lbl = edge_obj.get(label_key, None)
            if maybe_lbl:
                final_label = str(maybe_lbl)

        # leftover properties
        known_keys = { self.edge_field_map.get("oid"),
                       self.edge_field_map.get("source_id"),
                       self.edge_field_map.get("target_id"),
                       self.edge_field_map.get("start_time"),
                       self.edge_field_map.get("end_time"),
                       self.edge_field_map.get("label"),
                       self.edge_field_map.get("time_series_key","ts") }
        known_keys = { x for x in known_keys if x }  # remove None
        edge_properties = {}
        for k, v in edge_obj.items():
            if k not in known_keys:
                edge_properties[k] = v

        # ensure source/target exist
        if source_id not in self.hygraph.graph.nodes:
            print(f"   [WARN] Edge {external_id}: source node {source_id} not found.")
            return
        if target_id not in self.hygraph.graph.nodes:
            print(f"   [WARN] Edge {external_id}: target node {target_id} not found.")
            return

        # create or update
        existing_edge = None
        for u, v, key, data in self.hygraph.graph.edges(keys=True, data=True):
            if key == external_id:
                existing_edge = data["data"]
                break

        if not existing_edge:
            self.hygraph.add_pgedge(
                oid=external_id,
                source=source_id,
                target=target_id,
                label=final_label,
                start_time=start,
                end_time=end,
                properties=edge_properties
            )
        else:
            for kk, vv in edge_properties.items():
                existing_edge.add_static_property(kk, vv, self.hygraph)

        # handle time-series if "ts" or other key is present
        ts_obj_key = self.edge_field_map.get("time_series_key","ts")
        ts_obj = edge_obj.get(ts_obj_key,{})
        if isinstance(ts_obj, dict):
            self._process_edge_time_series(external_id, ts_obj)

    def _process_edge_time_series(self, external_id: str, ts_obj: Dict[str,Any]):
        edge_data = None
        for u, v, k, edata in self.hygraph.graph.edges(keys=True, data=True):
            if k == external_id:
                edge_data = edata["data"]
                break
        if not edge_data:
            return

        for ts_name, arr in ts_obj.items():
            if not isinstance(arr, list):
                continue
            tsid = f"{external_id}_{ts_name}"
            existing_ts = self.hygraph.time_series.get(tsid)
            if not existing_ts:
                timestamps = []
                values = []
                for rec in arr:
                    start_str = rec.get("Start", "")
                    val = rec.get("Value", 0)
                    parsed_start = simple_parse_date(start_str) or datetime.now()
                    timestamps.append(parsed_start)
                    values.append([val])
                metadata = TimeSeriesMetadata(owner_id=external_id, element_type="edge")
                new_ts = TimeSeries(tsid, timestamps, [ts_name], values, metadata)
                self.hygraph.time_series[tsid] = new_ts
                edge_data.add_temporal_property(ts_name, new_ts, self.hygraph)
            else:
                # update or append if needed
                pass
