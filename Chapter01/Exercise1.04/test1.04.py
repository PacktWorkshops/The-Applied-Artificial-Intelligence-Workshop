import unittest
import import_ipynb
import pandas as pd
import numpy as np
import pandas.testing as pd_testing
import numpy.testing as np_testing

from random import choice

class Test(unittest.TestCase):
	def setUp(self):
		import Exercise1_04
		self.exercises = Exercise1_04

		from random import choice

		self.combo_indices = [
			[0, 1, 2],
			[3, 4, 5],
			[6, 7, 8],
			[0, 3, 6],
			[1, 4, 7],
			[2, 5, 8],
			[0, 4, 8],
			[2, 4, 6]
		]

		self.EMPTY_SIGN = '.'
		self.AI_SIGN = 'X'
		self.OPPONENT_SIGN = 'O'

		def print_board(board):
			print(" ")
			print(' '.join(board[:3]))
			print(' '.join(board[3:6]))
			print(' '.join(board[6:]))
			print(" ")

		def opponent_move(board, row, column):
			index = 3 * (row - 1) + (column - 1)
			if board[index] == self.EMPTY_SIGN:
				return board[:index] + self.OPPONENT_SIGN + board[index + 1:]
			return board

		def all_moves_from_board(board, sign):
			move_list = []
			for i, v in enumerate(board):
				if v == self.EMPTY_SIGN:
					move_list.append(board[:i] + sign + board[i + 1:])
			return move_list

		def ai_move(board):
			return choice(all_moves_from_board(board, self.AI_SIGN))

		def game_won_by(board):
			for index in self.combo_indices:
				if board[index[0]] == board[index[1]] == board[index[2]] != self.EMPTY_SIGN:
					return board[index[0]]
			return self.EMPTY_SIGN

		def game_loop():
			board = self.EMPTY_SIGN * 9
			empty_cell_count = 9
			is_game_ended = False
			while empty_cell_count > 0 and not is_game_ended:
				if empty_cell_count % 2 == 1:
					board = ai_move(board)
				else:
					row = int(input('Enter row: '))
					col = int(input('Enter column: '))
					board = opponent_move(board, row, col)
				print_board(board)
				is_game_ended = game_won_by(board) != self.EMPTY_SIGN
				empty_cell_count = sum(1 for cell in board if cell == self.EMPTY_SIGN)
			print('Game has been ended.')

		def all_moves_from_board_list(board_list, sign):
			move_list = []
			for board in board_list:
				move_list.extend(all_moves_from_board(board, sign))
			return move_list

		def filter_wins(move_list, ai_wins, opponent_wins):
			for board in move_list:
				won_by = game_won_by(board)
				if won_by == self.AI_SIGN:
					ai_wins.append(board)
					move_list.remove(board)
				elif won_by == self.OPPONENT_SIGN:
					opponent_wins.append(board)
					move_list.remove(board)

		def count_possibilities():
			board = self.EMPTY_SIGN * 9
			move_list = [board]
			ai_wins = []
			opponent_wins = []
			for i in range(9):
				print('step ' + str(i) + '. Moves: ' + str(len(move_list)))
				sign = self.AI_SIGN if i % 2 == 0 else self.OPPONENT_SIGN
				move_list = all_moves_from_board_list(move_list, sign)
				filter_wins(move_list, ai_wins, opponent_wins)
			print('First player wins: ' + str(len(ai_wins)))
			print('Second player wins: ' + str(len(opponent_wins)))
			print('Draw', str(len(move_list)))
			print('Total', str(len(ai_wins) + len(opponent_wins) + len(move_list)))
			return len(ai_wins), len(opponent_wins), len(move_list), len(ai_wins) + len(opponent_wins) + len(move_list)

		def init_utility_matrix(board):
			return [0 if cell == self.EMPTY_SIGN else -1 for cell in board]

		def generate_add_score(utilities, i, j, k):
			def add_score(points):
				if utilities[i] >= 0:
					utilities[i] += points
				if utilities[j] >= 0:
					utilities[j] += points
				if utilities[k] >= 0:
					utilities[k] += points

			return add_score

		def utility_matrix(board):
			utilities = init_utility_matrix(board)
			for [i, j, k] in self.combo_indices:
				add_score = generate_add_score(utilities, i, j, k)
				triple = [board[i], board[j], board[k]]
				if triple.count(self.EMPTY_SIGN) == 1:
					if triple.count(self.AI_SIGN) == 2:
						add_score(1000)
					elif triple.count(self.OPPONENT_SIGN) == 2:
						add_score(100)
				elif triple.count(self.EMPTY_SIGN) == 2 and triple.count(self.AI_SIGN) == 1:
					add_score(10)
				elif triple.count(self.EMPTY_SIGN) == 3:
					add_score(1)
			return utilities

		def best_moves_from_board(board, sign):
			move_list = []
			utilities = utility_matrix(board)
			max_utility = max(utilities)
			for i, v in enumerate(board):
				if utilities[i] == max_utility:
					move_list.append(board[:i] + sign + board[i + 1:])
			return move_list

		def all_moves_from_board_list(board_list, sign):
			move_list = []
			get_moves = best_moves_from_board if sign == self.AI_SIGN else all_moves_from_board
			for board in board_list:
				move_list.extend(get_moves(board, sign))
			return move_list

		self.first_player, self.second_player, self.draw, self.total = count_possibilities()

	def test_combo_indices(self):
		np_testing.assert_array_equal(self.exercises.combo_indices, self.combo_indices)

	def test_EMPTY_SIGN(self):
		self.assertEqual(self.exercises.EMPTY_SIGN, self.EMPTY_SIGN)

	def test_AI_SIGN(self):
		self.assertEqual(self.exercises.AI_SIGN, self.AI_SIGN)

	def test_OPPONENT_SIGN(self):
		self.assertEqual(self.exercises.OPPONENT_SIGN, self.OPPONENT_SIGN)

	def test_first_player(self):
		self.assertEqual(self.exercises.first_player, self.first_player)

	def test_second_player(self):
		self.assertEqual(self.exercises.second_player, self.second_player)

	def test_draw(self):
		self.assertEqual(self.exercises.draw, self.draw)

	def test_total(self):
		self.assertEqual(self.exercises.total, self.total)

if __name__ == '__main__':
	unittest.main()
