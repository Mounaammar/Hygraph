import unittest
from datetime import datetime, timedelta
from hygraph_core.hygraph import HyGraph


def basic_trip_aggregator(hygraph, element_type, edge_id, attribute, dt):
    """
    For a given trip edge, return 1 if the client's age is under 18 at the given timestamp dt,
    otherwise return 0. Assumes the edge has a property 'year_of_birth'.
    """
    for u, v, key, data in hygraph.graph.edges(keys=True, data=True):
        if key == edge_id:
            edge_obj = data['data']
            yob = edge_obj.get_static_property('year_of_birth')
            if yob is not None:
                current_year = dt.year
                age = current_year - int(yob)
                return 1 if age < 18 else 0
            else:
                return 0
    return 0


class TestBasicEdgeAggregation(unittest.TestCase):
    def setUp(self):
        # Initialize a new HyGraph instance.
        self.hg = HyGraph()
        now = datetime(2023, 1, 1)
        # Add two station nodes.
        self.hg.add_pgnode("A", "StationA", now, now + timedelta(days=365))
        self.hg.add_pgnode("B", "StationB", now, now + timedelta(days=365))
        # Add three trip edges from A to B with different years of birth.
        # In 2023:
        #   - Trip1: year_of_birth = 2008  -> age 15 (under 18)
        #   - Trip2: year_of_birth = 2000  -> age 23 (not under 18)
        #   - Trip3: year_of_birth = 2009  -> age 14 (under 18)
        trip_times = [datetime(2023, 1, 5), datetime(2023, 1, 10), datetime(2023, 1, 15)]
        yobs = [2008, 2000, 2009]
        for i, (trip_time, yob) in enumerate(zip(trip_times, yobs), start=1):
            self.hg.add_pgedge(
                oid=f"trip{i}",
                source="A",
                target="B",
                label="Trip",
                start_time=trip_time,
                properties={"year_of_birth": yob}
            )

    def test_basic_edge_aggregation(self):
        # Define a query for edge-level time series generation.
        query_edge = {
            'element_type': 'edge',
            'edge_filter': lambda data: data['data'].label == 'Trip',
            'time_series_config': {
                'start_date': datetime(2023, 1, 1),
                'end_date': datetime(2023, 1, 31),
                'freq': 'D',
                'attribute': 'under18_trip_count',
                'aggregate_function': basic_trip_aggregator,
                'direction': 'both',
                'use_actual_timestamps': True
            },
            'aggregation': True,  # Enable grouping of edges into a super edge.
            'edge_settings': {
                'group_by': ['source', 'target', 'label']
            }
        }

        # Run the operator.
        self.hg.create_time_series_from_graph(query_edge)

        # The super edge is expected to have an ID "super_A_B_Trip".
        super_edge_id = "super_A_B_Trip"
        found = False
        for u, v, key, data in self.hg.graph.edges(keys=True, data=True):
            if key == super_edge_id:
                found = True
                super_edge = data['data']
                break
        self.assertTrue(found, f"Super edge {super_edge_id} was not created.")

        # Check that the dynamic property 'under18_trip_count' is attached.
        self.assertIn('under18_trip_count', super_edge.temporal_properties,
                      "Super edge should have the dynamic property 'under18_trip_count'.")
        ts_id = super_edge.get_temporal_property('under18_trip_count').get_id()
        print('her eis the time seires', ts_id)
        self.assertIn(ts_id, self.hg.time_series, "Time series ID not found in hygraph.time_series.")

        ts = self.hg.time_series[ts_id]
        # Sum the aggregated values across the time series.
        # Expect 1 (trip1) + 0 (trip2) + 1 (trip3) = 2 in total.
        total_under18_trips = sum(ts.data.values.flatten())
        self.assertEqual(total_under18_trips, 2,
                         f"Expected aggregated under18 trip count 2, got {total_under18_trips}")



class TestSubgraphTimeSeriesNodeFilter(unittest.TestCase):
    def setUp(self):
        self.hg = HyGraph()

        # 1) Create nodes with label="TestNode"
        now = datetime(2023, 1, 1)
        future = datetime(2100, 12, 31, 23, 59, 59)
        self.hg.add_pgnode("N1", "TestNode", now, future)
        self.hg.add_pgnode("N2", "TestNode", now, future)
        self.hg.add_pgnode("N3", "TestNode", now, future)
        self.hg.add_pgnode("N4", "TestNode", now, future)

        # Another node not of interest
        self.hg.add_pgnode("MISC", "OtherLabel", now, future)

        # 2) Timestamps for subgraph membership changes
        self.t1 = datetime(2023, 1, 2)
        self.t2 = datetime(2023, 1, 3)
        self.t3 = datetime(2023, 1, 4)

        # 3) Create subgraph "SG1"
        self.hg.add_subgraph(
            subgraph_id="SG1",
            label="MyDynamicSubgraph",
            start_time=self.t1,  # subgraph creation
            end_time=None
        )

        # 4) Manage membership
        #    At t1: N1, N2, N3 => subgraph size = 3
        self.hg.add_membership("N1", self.t1, ["SG1"], "node")
        self.hg.add_membership("N2", self.t1, ["SG1"], "node")
        self.hg.add_membership("N3", self.t1, ["SG1"], "node")

        #    At t2: add N4 => subgraph size=4
        self.hg.add_membership("N4", self.t2, ["SG1"], "node")

        #    At t3: remove membership for N2, N3 => subgraph size=2 (N1, N4 remain)
        self.hg.remove_membership("N2", self.t3, ["SG1"], "node")
        self.hg.remove_membership("N3", self.t3, ["SG1"], "node")

    def test_subgraph_with_node_filter(self):
        """
        Demonstrates using 'node_filter' in the subgraph query
        so we only consider nodes with label == 'TestNode'.
        We'll define an aggregator that counts the subgraph nodes
        currently in the subgraph view at each date,
        but the framework itself will skip any subgraph nodes
        that don't pass node_filter.
        """

        def subgraph_node_count_aggregator(hg, element_type, oid, attribute, dt):
            # aggregator that just returns subgraph_view.number_of_nodes()
            # The node_filter (in the query) ensures only nodes that pass the filter
            # appear in subgraph_view at dt.
            subgraph_view = hg.get_subgraph_at(oid, dt)
            return subgraph_view.number_of_nodes()

        # The node_filter ensures that only nodes with label == 'TestNode' appear in the subgraph
        # even if membership says "MISC" is included; it won't pass the node_filter =)
        def only_testnode_filter(subgraph_node_data):
            # subgraph_node_data is a dict from the HyGraph for each node
            # typically has 'label' or 'data'
            return subgraph_node_data.get('label', '') == 'TestNode'

        query_subgraph = {
            'element_type': 'subgraph',
            'subgraph_id': 'SG1',
            'node_filter': only_testnode_filter,
            'time_series_config': {
                'start_date': self.t1,
                'end_date': self.t3 + timedelta(days=1),
                'freq': 'D',
                'attribute': 'subgraph_size',
                'aggregate_function': subgraph_node_count_aggregator,
                'use_actual_timestamps': False,
            }
        }

        # Run the operator
        self.hg.create_time_series_from_graph(query_subgraph)

        # Retrieve the subgraph dynamic property
        sg_obj = self.hg.get_element('subgraph', 'SG1')
        self.assertIn('subgraph_size', sg_obj.temporal_properties,
                      "Subgraph SG1 should have 'subgraph_size' as a dynamic property.")

        # Convert to DataFrame
        ts_obj = sg_obj.temporal_properties['subgraph_size'].get_time_series()
        df_ts = ts_obj.data.to_dataframe('value').reset_index()  # time, variable, value
        print("\n[Subgraph TS with node_filter] 'subgraph_size':\n", df_ts)

        # We expect day by day:
        #   t1=2023-01-02 => 3
        #   t2=2023-01-03 => 4
        #   t3=2023-01-04 => 2
        expected_map = {
            self.t1.date(): 3,
            self.t2.date(): 4,
            self.t3.date(): 2,
        }

        for _, row in df_ts.iterrows():
            day = row['time'].date()
            val = row['value']
            if day in expected_map:
                self.assertEqual(val, expected_map[day],
                                 f"Subgraph size mismatch on day={day}: expected={expected_map[day]}, got={val}")

        # If we had included node "MISC" in subgraph membership,
        # the node_filter would have excluded it anyway, ensuring it never contributed.

if __name__ == '__main__':
    unittest.main()
