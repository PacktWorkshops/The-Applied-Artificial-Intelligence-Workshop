import unittest
import import_ipynb
import pandas as pd
import numpy as np
import pandas.testing as pd_testing
import numpy.testing as np_testing

from random import choice

class Test(unittest.TestCase):
	def setUp(self):
		import Exercise1_05
		self.exercises = Exercise1_05

		import math

		self.size = (7, 9)
		self.start = (5, 3)
		self.end = (6, 9)
		self.obstacles = {
			(3, 4), (3, 5), (3, 6), (3, 7), (3, 8),
			(4, 5),
			(5, 5), (5, 7), (5, 9),
			(6, 2), (6, 3), (6, 4), (6, 5), (6, 7),
			(7, 7)
		}

		def successors(state, visited_nodes):
			(row, col) = state
			(max_row, max_col) = self.size
			succ_states = []
			if row > 1:
				succ_states += [(row - 1, col)]
			if col > 1:
				succ_states += [(row, col - 1)]
			if row < max_row:
				succ_states += [(row + 1, col)]
			if col < max_col:
				succ_states += [(row, col + 1)]
			return [s for s in succ_states if s not in visited_nodes if s not in self.obstacles]

		def initialize_costs(size, start):
			(h, w) = size
			costs = [[math.inf] * w for i in range(h)]
			(x, y) = start
			costs[x - 1][y - 1] = 0
			return costs

		def update_costs(costs, current_node, successor_nodes):
			new_cost = costs[current_node[0] - 1][current_node[1] - 1] + 1
			for (x, y) in successor_nodes:
				costs[x - 1][y - 1] = min(costs[x - 1][y - 1], new_cost)

		def bfs_tree(node):
			nodes_to_visit = [node]
			visited_nodes = []
			costs = initialize_costs(self.size, self.start)
			while len(nodes_to_visit) > 0:
				current_node = nodes_to_visit.pop(0)
				visited_nodes.append(current_node)
				successor_nodes = successors(current_node, visited_nodes)
				update_costs(costs, current_node, successor_nodes)
				nodes_to_visit.extend(successor_nodes)
			return costs

		def bfs_tree_verbose(node):
			nodes_to_visit = [node]
			visited_nodes = []
			costs = initialize_costs(self.size, self.start)
			step_counter = 0
			while len(nodes_to_visit) > 0:
				step_counter += 1
				current_node = nodes_to_visit.pop(0)
				visited_nodes.append(current_node)
				successor_nodes = successors(current_node, visited_nodes)
				update_costs(costs, current_node, successor_nodes)
				nodes_to_visit.extend(successor_nodes)
				if current_node == self.end:
					print('End node has been reached in ', step_counter, ' steps')
					return costs
			return costs

		self.bfs = bfs_tree(self.start)
		self.bfs_v = bfs_tree_verbose(self.start)

	def test_size(self):
		self.assertEqual(self.exercises.size, self.size)

	def test_start(self):
		self.assertEqual(self.exercises.start, self.start)

	def test_end(self):
		self.assertEqual(self.exercises.end, self.end)

	def obstacles(self):
		self.assertEqual(self.exercises.obstacles, self.obstacles)

	def test_bfs(self):
		np_testing.assert_array_equal(self.exercises.bfs, self.bfs)

	def test_bfs_v(self):
		np_testing.assert_array_equal(self.exercises.bfs_v, self.bfs_v)

if __name__ == '__main__':
	unittest.main()
