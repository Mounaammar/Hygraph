"""
hygraph_universal_pipeline.py

A single entry point for the user that decides whether to use CSV loader
or JSON loader. Possibly processes both if needed.
"""

import os
from hygraph_core.hygraph import HyGraph
from .hygraph_csv_loader import HyGraphCSVLoader
from .hygraph_json_loader import HyGraphJSONLoader

class HyGraphUniversalPipeline:
    """
    A manager that picks the correct loader strategy based on user parameters
    or file extensions, then runs the pipeline.
    """
    def __init__(self, hygraph: HyGraph):
        self.hygraph = hygraph
        self.loader = None

    def configure_for_csv(
        self,
        nodes_folder: str,
        edges_folder: str,
        node_field_map: dict = None,
        edge_field_map: dict = None,
        max_rows_per_batch: int = 50_000
    ):
        self.loader = HyGraphCSVLoader(
            hygraph=self.hygraph,
            nodes_folder=nodes_folder,
            edges_folder=edges_folder,
            node_field_map=node_field_map,
            edge_field_map=edge_field_map,
            max_rows_per_batch=max_rows_per_batch
        )

    def configure_for_json(
        self,
        node_json_path: str,
        edge_json_path: str,
        membership_json_path: str = None,
        node_field_map: dict = None,
        edge_field_map: dict = None,
    ):
        self.loader = HyGraphJSONLoader(
            hygraph=self.hygraph,
            node_json_path=node_json_path,
            edge_json_path=edge_json_path,
            node_field_map=node_field_map,
            edge_field_map=edge_field_map
        )

    def run_pipeline(self):
        """
        Calls the chosen loader's run_all() if we have one configured,
        or does nothing if none is configured.
        """
        if not self.loader:
            print("No loader configured! Please configure_for_csv or configure_for_json.")
            return
        self.loader.run_pipeline()

        # Could display or do final post-processing
        #self.hygraph.display()
        print("HyGraph pipeline run complete!")
