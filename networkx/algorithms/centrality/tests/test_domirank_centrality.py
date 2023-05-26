import math

import pytest

import networkx as nx


class TestDomiRankCentrality:
    def test_K5(self):
        """DomiRank centrality: K5"""
        G = nx.complete_graph(5)
        sigma = 0.1
        dr = nx.domirank_centrality(G, sigma)
        return dr
