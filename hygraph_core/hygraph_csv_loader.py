"""
hygraph_csv_loader.py

An advanced ETL pipeline for loading large CSV files into HyGraph,
using Polars for streaming and chunk-based reading, with schema validation.
Also supports user-defined field mappings (node_field_map, edge_field_map)
and time-series columns (node_ts_columns, edge_ts_columns).

Requires:
    pip install polars
"""

import os
import polars as pl
from datetime import datetime
from typing import Optional, Any, Dict, List

from hygraph_core.hygraph import HyGraph
from hygraph_core.constraints import parse_datetime
from hygraph_core.timeseries_operators import TimeSeries, TimeSeriesMetadata

# A far-future date for open-ended intervals
FAR_FUTURE_DATE = datetime(2100, 12, 31, 23, 59, 59)

#############################
#   SCHEMA DEFINITIONS     #
#############################

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

SUBGRAPHS_SCHEMA = {
    "id": pl.Utf8,
    "start_time": pl.Utf8,
    "end_time": pl.Utf8,
    # Additional subgraph columns
}

MEMBERSHIP_SCHEMA = {
    "id": pl.Utf8,
    "subgraph_id": pl.Utf8,
    "timestamp": pl.Utf8,   # We'll parse the date ourselves
    "action": pl.Utf8
}


def parse_iso_date(date_str: str) -> Optional[datetime]:
    """
    Minimal date parser for ISO-like strings, e.g. "2024-05-16T00:00:00".
    Adjust if your CSV uses a different date format.
    """
    if not date_str:
        return None
    try:
        return datetime.fromisoformat(date_str)
    except ValueError:
        return None


class HyGraphCSVLoader:
    """
    A specialized ETL pipeline that can read large CSV files via Polars,
    create or update HyGraph nodes, edges, subgraphs, and handle membership.
    With schema validation to ensure CSV columns match expectations.
    Also supports user-defined field mappings and simple time-series columns.
    """

    def __init__(
        self,
        hygraph: HyGraph,
        nodes_folder: str,
        edges_folder: str,
        subgraphs_folder: str,
        edges_membership_path: str,
        nodes_membership_path: str,
        max_rows_per_batch: int = 50_000,
        # Below are new:
        node_field_map: Dict[str, str] = None,
        edge_field_map: Dict[str, str] = None,
        node_ts_columns: List[str] = None,
        edge_ts_columns: List[str] = None,
    ):
        """
        :param hygraph: An instance of HyGraph where data is loaded.
        :param nodes_folder: Directory containing node CSVs.
        :param edges_folder: Directory containing edge CSVs.
        :param subgraphs_folder: Directory containing subgraph CSVs.
        :param edges_membership_path: CSV file path for edges membership info.
        :param nodes_membership_path: CSV file path for nodes membership info.
        :param max_rows_per_batch: Number of rows to process per chunk/batch.

        :param node_field_map: A dictionary telling us how to map CSV columns
                               to internal fields. Example:
                               {
                                 "oid": "id",
                                 "start_time": "start_time",
                                 "end_time": "end_time"
                               }
                               If omitted, we assume "id","start_time","end_time"
                               directly.

        :param edge_field_map: Similar for edges. Example:
                               {
                                 "oid": "id",
                                 "source_id": "source_id",
                                 "target_id": "target_id",
                                 "start_time": "start_time",
                                 "end_time": "end_time"
                               }

        :param node_ts_columns: List of CSV columns for node time-series data.
                                Each row is interpreted as one data point,
                                using the row's 'start_time' as the time dimension.
        :param edge_ts_columns: Similarly for edges.

        Example usage:
            node_field_map = {"oid":"id","start_time":"start_time","end_time":"end_time"}
            edge_field_map = {"oid":"id","source_id":"source_id","target_id":"target_id","start_time":"start_time"}
            node_ts_columns = ["num_bikes","num_docks"]
            edge_ts_columns = ["num_rides","member_rides"]
        """
        self.hygraph = hygraph
        self.nodes_folder = nodes_folder
        self.edges_folder = edges_folder
        self.subgraphs_folder = subgraphs_folder
        self.edges_membership_path = edges_membership_path
        self.nodes_membership_path = nodes_membership_path
        self.max_rows_per_batch = max_rows_per_batch

        self.node_field_map = node_field_map or {}
        self.edge_field_map = edge_field_map or {}
        self.node_ts_columns = node_ts_columns or []
        self.edge_ts_columns = edge_ts_columns or []

    ########################
    #     MAIN PIPELINE    #
    ########################

    def run_pipeline(self):
        """
        Master pipeline method:
          1. Load all nodes (chunked)
          2. Load all edges (chunked)
          3. Load all subgraphs (chunked)
          4. Process membership for edges
          5. Process membership for nodes
          6. Print final status
        """
        print("========== Starting ETL Pipeline (with Schema) ==========")
        self.load_all_nodes()
        self.load_all_edges()
        self.load_all_subgraphs()

        # For membership, typically smaller CSVs—but we still do chunked to be safe
        self.process_membership_data(self.edges_membership_path, element_type="edge")
        self.process_membership_data(self.nodes_membership_path, element_type="node")

        # Optionally, you might finalize or do post-processing
        self.finalize_pipeline()
        print("========== ETL Pipeline Complete ==========")

    def finalize_pipeline(self):
        print("\nFinalizing the pipeline... current HyGraph state:")
        self.hygraph.display()
        # Additional steps (e.g., indexing, queries) can be done here.

    ########################
    #       LOAD NODES     #
    ########################

    def load_all_nodes(self):
        """
        Load all .csv files in `self.nodes_folder`, chunk by chunk using Polars.
        Each CSV is assumed to define a "label" (derived from filename).
        """
        node_files = [
            f for f in os.listdir(self.nodes_folder)
            if f.endswith(".csv")
        ]
        for file_name in node_files:
            label = file_name.replace(".csv", "")
            file_path = os.path.join(self.nodes_folder, file_name)
            self.load_nodes_from_csv(file_path, label)

    def load_nodes_from_csv(self, csv_path: str, label: str):
        """
        Stream a single CSV of nodes in chunked fashion using Polars,
        with the NODES_SCHEMA for validation.
        """
        print(f"\n[Nodes] Loading from {csv_path} with label={label}")
        # Apply schema with dtypes
        scan = pl.scan_csv(csv_path, dtypes=NODES_SCHEMA)

        offset = 0
        batch_idx = 0
        while True:
            df = scan.limit(self.max_rows_per_batch).offset(offset).collect()
            if df.height == 0:
                break  # No more rows
            batch_idx += 1
            print(f"   -> Processing Node Batch #{batch_idx} with {df.height} rows (offset={offset})")
            self._process_node_batch(df, label)
            offset += df.height

    def _process_node_batch(self, df: pl.DataFrame, label: str):
        """
        For each row, insert or update a PGNode in HyGraph, applying the node_field_map
        for ID, start_time, end_time, plus any time-series columns.
        """
        for row in df.iter_rows(named=True):
            self._process_node_record(row, label)

    def _process_node_record(self, row: Dict[str,Any], label: str):
        # Extract columns using node_field_map
        # Example: "oid" -> "id" means node_field_map["oid"] = "id"
        oid_col = self.node_field_map.get("oid", "id")
        start_col = self.node_field_map.get("start_time", "start_time")
        end_col = self.node_field_map.get("end_time", "end_time")

        external_id = str(row.get(oid_col, "")) or f"node_{id(row)}"

        start_time = self._safe_parse_date(row.get(start_col), default=datetime.now())
        end_time = self._safe_parse_date(row.get(end_col), default=FAR_FUTURE_DATE)

        # Build leftover properties from columns not in [oid_col, start_col, end_col]
        known_cols = {oid_col, start_col, end_col}
        props = {}
        for k, v in row.items():
            # Skip if it's in known_cols or if it's in node_ts_columns
            if k in known_cols or k in self.node_ts_columns:
                continue
            props[k] = v

        # Insert or update node
        if external_id not in self.hygraph.graph.nodes:
            self.hygraph.add_pgnode(
                oid=external_id,
                label=label,
                start_time=start_time,
                end_time=end_time,
                properties=props
            )
        else:
            existing_node = self.hygraph.graph.nodes[external_id]["data"]
            for kk, vv in props.items():
                existing_node.add_static_property(kk, vv, self.hygraph)

        # If we have time-series columns, interpret them as values at "start_time"
        if self.node_ts_columns:
            self._process_node_time_series_columns(external_id, row, start_time)

    def _process_node_time_series_columns(self, external_id: str, row: Dict[str, Any], timestamp: datetime):
        """
        Each CSV row is one data point for each of node_ts_columns, using `timestamp`.
        """
        node_data = self.hygraph.graph.nodes[external_id]["data"]
        for col_name in self.node_ts_columns:
            if col_name not in row:
                continue
            val = row[col_name]
            tsid = f"{external_id}_{col_name}"  # unique timeseries ID
            existing_ts = self.hygraph.time_series.get(tsid)
            if not existing_ts:
                # create a new timeseries
                metadata = TimeSeriesMetadata(owner_id=external_id, element_type="node")
                new_ts = TimeSeries(
                    tsid=tsid,
                    timestamps=[timestamp],
                    variables=[col_name],
                    data=[[val]],  # Nx1
                    metadata=metadata
                )
                self.hygraph.time_series[tsid] = new_ts
                node_data.add_temporal_property(col_name, new_ts, self.hygraph)
            else:
                # append or update
                if existing_ts.has_timestamp(timestamp):
                    existing_ts.update_value_at_timestamp(timestamp, val, variable_name=col_name)
                else:
                    existing_ts.append_data(timestamp, val)

    ########################
    #       LOAD EDGES     #
    ########################

    def load_all_edges(self):
        edge_files = [
            f for f in os.listdir(self.edges_folder)
            if f.endswith(".csv")
        ]
        for file_name in edge_files:
            label = file_name.replace(".csv", "")
            file_path = os.path.join(self.edges_folder, file_name)
            self.load_edges_from_csv(file_path, label)

    def load_edges_from_csv(self, csv_path: str, label: str):
        """
        Read edges with the EDGES_SCHEMA to ensure correct columns and types.
        """
        print(f"\n[Edges] Loading from {csv_path} with label={label}")
        scan = pl.scan_csv(csv_path, dtypes=EDGES_SCHEMA)

        offset = 0
        batch_idx = 0
        while True:
            df = scan.limit(self.max_rows_per_batch).offset(offset).collect()
            if df.height == 0:
                break
            batch_idx += 1
            print(f"   -> Processing Edge Batch #{batch_idx} with {df.height} rows (offset={offset})")
            self._process_edge_batch(df, label)
            offset += df.height

    def _process_edge_batch(self, df: pl.DataFrame, label: str):
        for row in df.iter_rows(named=True):
            self._process_edge_record(row, label)

    def _process_edge_record(self, row: Dict[str,Any], label: str):
        # Use edge_field_map to interpret CSV columns
        oid_col = self.edge_field_map.get("oid", "id")
        src_col = self.edge_field_map.get("source_id", "source_id")
        tgt_col = self.edge_field_map.get("target_id", "target_id")
        st_col = self.edge_field_map.get("start_time", "start_time")
        ed_col = self.edge_field_map.get("end_time", "end_time")

        edge_id = str(row.get(oid_col, "")) or f"edge_{id(row)}"
        source_id = str(row.get(src_col, ""))
        target_id = str(row.get(tgt_col, ""))
        start_time = self._safe_parse_date(row.get(st_col), default=datetime.now())
        end_time = self._safe_parse_date(row.get(ed_col), default=FAR_FUTURE_DATE)

        # leftover properties
        known_cols = {oid_col, src_col, tgt_col, st_col, ed_col}
        props = {}
        for k, v in row.items():
            if k in known_cols or k in self.edge_ts_columns:
                continue
            props[k] = v

        if source_id not in self.hygraph.graph.nodes:
            print(f"      [WARN] Skipping Edge {edge_id}: Source {source_id} not found.")
            return
        if target_id not in self.hygraph.graph.nodes:
            print(f"      [WARN] Skipping Edge {edge_id}: Target {target_id} not found.")
            return

        # Check if edge already in graph
        existing_edge = None
        for u, v, key, data in self.hygraph.graph.edges(keys=True, data=True):
            if key == edge_id:
                existing_edge = data["data"]
                break

        if not existing_edge:
            self.hygraph.add_pgedge(
                oid=edge_id,
                source=source_id,
                target=target_id,
                label=label,
                start_time=start_time,
                end_time=end_time,
                properties=props
            )
        else:
            for kk, val in props.items():
                existing_edge.add_static_property(kk, val, self.hygraph)

        # If we have time-series columns, interpret them as values at "start_time"
        if self.edge_ts_columns:
            self._process_edge_time_series_columns(edge_id, row, start_time)

    def _process_edge_time_series_columns(self, edge_id: str, row: Dict[str,Any], timestamp: datetime):
        # find the edge in the graph
        edge_data = None
        for u, v, k, edata in self.hygraph.graph.edges(keys=True, data=True):
            if k == edge_id:
                edge_data = edata["data"]
                break
        if not edge_data:
            return

        for col_name in self.edge_ts_columns:
            if col_name not in row:
                continue
            val = row[col_name]
            tsid = f"{edge_id}_{col_name}"
            existing_ts = self.hygraph.time_series.get(tsid)
            if not existing_ts:
                metadata = TimeSeriesMetadata(owner_id=edge_id, element_type="edge")
                new_ts = TimeSeries(
                    tsid=tsid,
                    timestamps=[timestamp],
                    variables=[col_name],
                    data=[[val]],
                    metadata=metadata
                )
                self.hygraph.time_series[tsid] = new_ts
                edge_data.add_temporal_property(col_name, new_ts, self.hygraph)
            else:
                if existing_ts.has_timestamp(timestamp):
                    existing_ts.update_value_at_timestamp(timestamp, val, variable_name=col_name)
                else:
                    existing_ts.append_data(timestamp, val)

    ########################
    #    LOAD SUBGRAPHS    #
    ########################

    def load_all_subgraphs(self):
        subgraph_files = [
            f for f in os.listdir(self.subgraphs_folder)
            if f.endswith(".csv")
        ]
        for file_name in subgraph_files:
            label = file_name.replace(".csv", "")
            file_path = os.path.join(self.subgraphs_folder, file_name)
            self.load_subgraphs_from_csv(file_path, label)

    def load_subgraphs_from_csv(self, csv_path: str, label: str):
        print(f"\n[Subgraphs] Loading from {csv_path} with label={label}")
        scan = pl.scan_csv(csv_path, dtypes=SUBGRAPHS_SCHEMA)

        offset = 0
        batch_idx = 0
        while True:
            df = scan.limit(self.max_rows_per_batch).offset(offset).collect()
            if df.height == 0:
                break
            batch_idx += 1
            print(f"   -> Processing Subgraph Batch #{batch_idx} with {df.height} rows (offset={offset})")
            self._process_subgraph_batch(df, label)
            offset += df.height

    def _process_subgraph_batch(self, df: pl.DataFrame, label: str):
        for row in df.iter_rows(named=True):
            subgraph_id = str(row["id"])
            start_time = self._safe_parse_date(row.get("start_time"), default=datetime.now())
            end_time = self._safe_parse_date(row.get("end_time"), default=FAR_FUTURE_DATE)

            # Additional properties
            props = {
                k: v for k, v in row.items()
                if k not in ["id", "start_time", "end_time"]
            }
            props["label"] = label

            self.hygraph.add_subgraph(
                subgraph_id=subgraph_id,
                label=label,
                properties=props,
                start_time=start_time,
                end_time=end_time
            )

    ########################
    #   MEMBERSHIP LOGIC   #
    ########################

    def process_membership_data(self, membership_csv: str, element_type: str):
        """
        A membership approach: chunk-read the CSV with Polars,
        parse each row -> call hygraph.add_membership or remove_membership.
        """
        if not os.path.isfile(membership_csv):
            print(f"[Membership] File not found: {membership_csv} (skipping).")
            return

        # Use MEMBERSHIP_SCHEMA
        print(f"\n[Membership] Processing {membership_csv} for element_type={element_type}")
        scan = pl.scan_csv(membership_csv, dtypes=MEMBERSHIP_SCHEMA)

        offset = 0
        batch_idx = 0
        while True:
            df = scan.limit(self.max_rows_per_batch).offset(offset).collect()
            if df.height == 0:
                break
            batch_idx += 1
            print(f"   -> Membership Batch #{batch_idx} with {df.height} rows")
            self._process_membership_batch(df, element_type)
            offset += df.height

    def _process_membership_batch(self, df: pl.DataFrame, element_type: str):
        # Sort by timestamp for chronological updates
        df = df.sort(by="timestamp")

        for row in df.iter_rows(named=True):
            external_ids = str(row["id"]).split()
            subgraph_ids = str(row["subgraph_id"]).split()
            timestamp = self._safe_parse_date(row["timestamp"], default=datetime.now())
            action = str(row["action"]).strip().lower()

            for external_id in external_ids:
                external_id = external_id.strip()
                if element_type == "node":
                    if external_id not in self.hygraph.graph.nodes:
                        print(f"   [WARN] Node {external_id} not found, skipping membership.")
                        continue
                elif element_type == "edge":
                    if not self._edge_exists(external_id):
                        print(f"   [WARN] Edge {external_id} not found, skipping membership.")
                        continue
                else:
                    print(f"   [WARN] Unrecognized element_type={element_type}, skipping membership.")
                    continue

                if action == "add":
                    self.hygraph.add_membership(external_id, timestamp, subgraph_ids, element_type)
                elif action == "remove":
                    self.hygraph.remove_membership(external_id, timestamp, subgraph_ids, element_type)
                else:
                    print(f"   [WARN] Unknown action={action} for {element_type}={external_id}")

    def _edge_exists(self, edge_id: str) -> bool:
        """
        Check if an edge with key=edge_id is in self.hygraph.
        """
        for _, _, k, _ in self.hygraph.graph.edges(keys=True, data=True):
            if k == edge_id:
                return True
        return False

    ########################
    #      UTILITIES       #
    ########################

    def _safe_parse_date(self, val: Any, default: Optional[datetime] = None) -> datetime:
        """
        Try to parse a date/datetime from a polars cell. Fallback to `default`.
        """
        if not val:
            return default if default else datetime.now()
        if isinstance(val, datetime):
            return val
        # Attempt parsing
        parsed = parse_datetime(str(val))
        return parsed if parsed else (default if default else datetime.now())
