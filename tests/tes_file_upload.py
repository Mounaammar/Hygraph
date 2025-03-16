import os
import json
import unittest
import tempfile
from datetime import datetime

# Import HyGraph and the universal pipeline from your hygraph_core package.
from hygraph_core.hygraph import HyGraph
from hygraph_core.hygraph_universal_pipeline import HyGraphUniversalPipeline

class TestHyGraphJSONLoader(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory to hold our sample JSON files.
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_dir = self.temp_dir.name

        # Create nodes directory and write a sample JSON file.
        self.nodes_dir = os.path.join(self.base_dir, "nodes")
        os.makedirs(self.nodes_dir, exist_ok=True)
        self.node_file = os.path.join(self.nodes_dir, "minimal_nodes.json")
        sample_nodes = [
            {
                "station_id": "N001",
                "start": "2024-01-01T00:00:00",
                "end": "2025-12-31T23:59:59",
                "name": "Test Station A"
            },
            {
                "station_id": "N002",
                "start": "2024-06-01T10:00:00",
                "end": "",
                "name": "Test Station B"
            }
        ]
        with open(self.node_file, "w") as f:
            json.dump(sample_nodes, f)

        # Create edges directory and write a sample JSON file.
        self.edges_dir = os.path.join(self.base_dir, "edges")
        os.makedirs(self.edges_dir, exist_ok=True)
        self.edge_file = os.path.join(self.edges_dir, "minimal_edges.json")
        sample_edges = [
            {
                "id": "E001",
                "from": "N001",
                "to": "N002",
                "start": "2025-01-01T09:00:00",
                "end": "",
                "num_rides": 25
            },
            {
                "id": "E002",
                "from": "N002",
                "to": "N001",
                "start": "2025-02-01T12:00:00",
                "end": "2026-01-01T00:00:00",
                "classic_rides": 10,
                "electric_rides": 5
            }
        ]
        with open(self.edge_file, "w") as f:
            json.dump(sample_edges, f)

        # Create a fresh HyGraph instance and a universal pipeline.
        self.hygraph = HyGraph()
        self.pipeline = HyGraphUniversalPipeline(self.hygraph)

        # Define the field mappings for nodes and edges.
        self.node_field_map = {
            "oid": "station_id",  # maps CSV/JSON 'station_id' to node internal ID
            "start_time": "start",
            "end_time": "end"
            # Optionally, "labels": "labels", "time_series_key": "ts"
        }
        self.edge_field_map = {
            "oid": "id",  # maps JSON 'id' to edge internal ID
            "source_id": "from",
            "target_id": "to",
            "start_time": "start",
            "end_time": "end"
            # Optionally, "label": "label", "time_series_key": "ts"
        }

        # Configure the pipeline for JSON ingestion.
        self.pipeline.configure_for_json(
            node_json_path=self.nodes_dir,
            edge_json_path=self.edges_dir,
            node_field_map=self.node_field_map,
            edge_field_map=self.edge_field_map
        )

    def tearDown(self):
        # Clean up the temporary directory.
        self.temp_dir.cleanup()

    def test_json_etl_pipeline(self):
        # Run the JSON ETL pipeline.
        self.pipeline.run_pipeline()

        # Verify that nodes and edges were imported.
        node_count = len(self.hygraph.graph.nodes)
        edge_count = len(self.hygraph.graph.edges)
        self.assertEqual(node_count, 2, f"Expected 2 nodes, got {node_count}")
        self.assertEqual(edge_count, 2, f"Expected 2 edges, got {edge_count}")

        # Verify that the nodes have the expected properties.
        self.assertIn("N001", self.hygraph.graph.nodes)
        node_data_1 = self.hygraph.graph.nodes["N001"]["data"]
        self.assertEqual(node_data_1.get_static_property("name"), "Test Station A")

        self.assertIn("N002", self.hygraph.graph.nodes)
        node_data_2 = self.hygraph.graph.nodes["N002"]["data"]
        self.assertEqual(node_data_2.get_static_property("name"), "Test Station B")

        # Verify that at least one expected edge exists.
        all_edges = list(self.hygraph.graph.edges(keys=True, data=True))
        found_e001 = any(k == "E001" for _, _, k, _ in all_edges)
        self.assertTrue(found_e001, "Edge E001 not found")

        print("test_json_etl_pipeline succeeded: 2 nodes and 2 edges loaded as expected!")

if __name__ == '__main__':
    unittest.main()
