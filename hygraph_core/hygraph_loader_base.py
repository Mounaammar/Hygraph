"""
hygraph_loader_base.py

An abstract base class defining the interface for loaders that parse data (CSV or JSON)
and insert/update HyGraph.

 +------------------------+
 |  HyGraphFileManager   |
 |  (Unifying pipeline)  |
 +----------+------------+
            | (decides CSV or JSON at runtime)
            |
     +-------v----------------+                  +---------------------------------+
     | HyGraphCSVLoader       |                  | HyGraphJSONLoader               |
     | - load_nodes()         |                  | - load_nodes()                  |
     | - load_edges()         |  implements -->  | - load_edges()                  |
     | - (membership, etc.)   |                  | - (membership, etc.)            |
     +------------------------+                  +---------------------------------+
                   \                                      /
                    \                                    /
                     \---->  HyGraph (the final graph) <----

"""

from abc import ABC, abstractmethod
from hygraph_core.hygraph import HyGraph

class HyGraphLoaderBase(ABC):
    """
    An abstract base loader specifying the interface that CSV or JSON loaders must implement.
    """

    def __init__(self, hygraph: HyGraph):
        self.hygraph = hygraph

    @abstractmethod
    def load_nodes(self):
        """Load (or update) nodes in HyGraph from the data source (CSV or JSON)."""

    @abstractmethod
    def load_edges(self):
        """Load (or update) edges in HyGraph from the data source."""

    @abstractmethod
    def run_all(self):
        """Convenient method to run the entire pipeline for nodes & edges."""
