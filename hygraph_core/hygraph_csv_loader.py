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
from hygraph_core.constraints import parse_datetime  # your custom parse_datetime function
from hygraph_core.timeseries_operators import TimeSeries, TimeSeriesMetadata
from hygraph_core.utils import timeit

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

class HyGraphCSVLoader:
    """
    A specialized ETL pipeline that reads large CSV files via Polars,
    creates/updates HyGraph nodes and edges, and handles time-series columns.
    """

    def __init__(
        self,
        hygraph: HyGraph,
        nodes_folder: str,
        edges_folder: str,
        max_rows_per_batch: int = 50_000,
        node_field_map: Dict[str, str] = None,
        edge_field_map: Dict[str, str] = None,
        node_ts_columns: List[str] = None,
        edge_ts_columns: List[str] = None,
    ):
        self.hygraph = hygraph
        self.nodes_folder = nodes_folder
        self.edges_folder = edges_folder
        self.max_rows_per_batch = max_rows_per_batch
        self.node_field_map = node_field_map or {}
        self.edge_field_map = edge_field_map or {}
        self.node_ts_columns = node_ts_columns or []
        self.edge_ts_columns = edge_ts_columns or []

    ########################
    #     MAIN PIPELINE    #
    ########################
    @timeit
    def run_pipeline(self):
        print("========== Starting ETL Pipeline (with Schema) ==========")
        self.load_all_nodes()
        self.load_all_edges()
        self.finalize_pipeline()
        print("========== ETL Pipeline Complete ==========")

    def finalize_pipeline(self):
        print("\nFinalizing the pipeline... current HyGraph state:")


    ########################
    #       LOAD NODES     #
    ########################
    @timeit
    def load_all_nodes(self):
        node_files = [f for f in os.listdir(self.nodes_folder) if f.endswith(".csv")]
        for file_name in node_files:
            label = file_name.replace(".csv", "")
            file_path = os.path.join(self.nodes_folder, file_name)
            self.load_nodes_from_csv(file_path, label)

    @timeit
    def load_nodes_from_csv(self, csv_path: str, label: str):
        print(f"\n[Nodes] Loading from {csv_path} with label={label}")
        scan = pl.scan_csv(csv_path, dtypes=NODES_SCHEMA)
        offset = 0
        batch_idx = 0
        while True:
            df = scan.slice(offset, self.max_rows_per_batch).collect()
            if df.height == 0:
                break
            batch_idx += 1
            print(f"   -> Processing Node Batch #{batch_idx} with {df.height} rows (offset={offset})")
            self._process_node_batch(df, label)
            offset += df.height

    def _process_node_batch(self, df: pl.DataFrame, label: str):
        for row in df.iter_rows(named=True):
            self._process_node_record(row, label)


    def _process_node_record(self, row: Dict[str, Any], label: str):
        oid_col = self.node_field_map.get("oid", "id")
        start_col = self.node_field_map.get("start_time", "start_time")
        end_col = self.node_field_map.get("end_time", "end_time")

        external_id = str(row.get(oid_col, "")) or f"node_{id(row)}"
        # Parse datetime fields using _safe_parse_date
        start_time = self._safe_parse_date(row.get(start_col), default=datetime.now())
        end_time = self._safe_parse_date(row.get(end_col), default=FAR_FUTURE_DATE)

        known_cols = {oid_col, start_col, end_col}
        props = {k: v for k, v in row.items() if k not in known_cols and k not in self.node_ts_columns}

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

        if self.node_ts_columns:
            self._process_node_time_series_columns(external_id, row, start_time)

    def _process_node_time_series_columns(self, external_id: str, row: Dict[str, Any], timestamp: datetime):
        node_data = self.hygraph.graph.nodes[external_id]["data"]
        for col_name in self.node_ts_columns:
            if col_name not in row:
                continue
            val = row[col_name]
            tsid = f"{external_id}_{col_name}"
            existing_ts = self.hygraph.time_series.get(tsid)
            if not existing_ts:
                metadata = TimeSeriesMetadata(owner_id=external_id, element_type="node")
                new_ts = TimeSeries(tsid=tsid, timestamps=[timestamp], variables=[col_name], data=[[val]], metadata=metadata)
                self.hygraph.time_series[tsid] = new_ts
                node_data.add_temporal_property(col_name, new_ts, self.hygraph)
            else:
                if existing_ts.has_timestamp(timestamp):
                    existing_ts.update_value_at_timestamp(timestamp, val, variable_name=col_name)
                else:
                    existing_ts.append_data(timestamp, val)

    ########################
    #       LOAD EDGES     #
    ########################
    @timeit
    def load_all_edges(self):
        edge_files = [f for f in os.listdir(self.edges_folder) if f.endswith(".csv")]
        for file_name in edge_files:
            label = file_name.replace(".csv", "")
            file_path = os.path.join(self.edges_folder, file_name)
            self.load_edges_from_csv(file_path, label)

    @timeit
    def load_edges_from_csv(self, csv_path: str, label: str):
        print(f"\n[Edges] Loading from {csv_path} with label={label}")
        scan = pl.scan_csv(csv_path, dtypes=EDGES_SCHEMA)
        offset = 0
        batch_idx = 0
        while True:
            df = scan.slice(offset, self.max_rows_per_batch).collect()
            if df.height == 0:
                break
            batch_idx += 1
            print(f"   -> Processing Edge Batch #{batch_idx} with {df.height} rows (offset={offset})")
            self._process_edge_batch(df, label)
            offset += df.height

    def _process_edge_batch(self, df: pl.DataFrame, label: str):
        for row in df.iter_rows(named=True):
            self._process_edge_record(row, label)


    def _process_edge_record(self, row: Dict[str, Any], label: str):
        oid_col = self.edge_field_map.get("oid", "id")
        src_col = self.edge_field_map.get("source_id", "source_id")
        tgt_col = self.edge_field_map.get("target_id", "target_id")
        st_col = self.edge_field_map.get("start_time", "start_time")
        ed_col = self.edge_field_map.get("end_time", "end_time")

        edge_id = str(row.get(oid_col, "")) or f"edge_{id(row)}"
        source_id = str(row.get(src_col, ""))
        target_id = str(row.get(tgt_col, ""))
        # Parse datetime fields for proper comparisons
        start_time = self._safe_parse_date(row.get(st_col), default=datetime.now())
        end_time = self._safe_parse_date(row.get(ed_col), default=FAR_FUTURE_DATE)

        known_cols = {oid_col, src_col, tgt_col, st_col, ed_col}
        props = {k: v for k, v in row.items() if k not in known_cols and k not in self.edge_ts_columns}

        if source_id not in self.hygraph.graph.nodes:
            print(f"      [WARN] Skipping Edge {edge_id}: Source {source_id} not found.")
            return
        if target_id not in self.hygraph.graph.nodes:
            print(f"      [WARN] Skipping Edge {edge_id}: Target {target_id} not found.")
            return

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

        if self.edge_ts_columns:
            self._process_edge_time_series_columns(edge_id, row, start_time)

    def _process_edge_time_series_columns(self, edge_id: str, row: Dict[str, Any], timestamp: datetime):
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
                new_ts = TimeSeries(tsid=tsid, timestamps=[timestamp], variables=[col_name], data=[[val]], metadata=metadata)
                self.hygraph.time_series[tsid] = new_ts
                edge_data.add_temporal_property(col_name, new_ts, self.hygraph)
            else:
                if existing_ts.has_timestamp(timestamp):
                    existing_ts.update_value_at_timestamp(timestamp, val, variable_name=col_name)
                else:
                    existing_ts.append_data(timestamp, val)

    ########################
    #      UTILITIES       #
    ########################

    def _safe_parse_date(self, val: Any, default: Optional[datetime] = None) -> datetime:
        """
        Try to parse a date/datetime from a string using multiple formats.
        If no format matches, return the default value (or current datetime if not provided).
        """
        if not val:
            return default if default else datetime.now()

        # List of datetime formats to try
        formats = [
            "%Y-%m-%dT%H:%M:%S.%f",  # e.g., "2024-01-22T18:43:19.012000"
            "%Y-%m-%dT%H:%M:%S",  # e.g., "2015-03-01T00:00:00"
            "%Y-%m-%d %H:%M:%S",  # e.g., "2015-03-01 00:00:00"
            "%m/%d/%y",  # e.g., "01/15/24"
            "%m/%d/%Y",  # e.g., "01/15/2024"
        ]

        for fmt in formats:
            try:
                return datetime.strptime(val, fmt)
            except ValueError:
                continue

        print(f"Failed to parse datetime: {val}")
        return default if default else datetime.now()
