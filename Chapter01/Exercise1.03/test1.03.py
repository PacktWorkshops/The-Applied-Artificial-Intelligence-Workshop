import unittest
import import_ipynb
import pandas as pd
import numpy as np
import pandas.testing as pd_testing
import numpy.testing as np_testing

from random import choice

class Test(unittest.TestCase):
	def setUp(self):
		import Exercise1_03
		self.exercises = Exercise1_03

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

		def player_can_win(board, sign):
			next_moves = all_moves_from_board(board, sign)
			for next_move in next_moves:
				if game_won_by(next_move) == sign:
					return True
			return False

		def ai_move(board):
			new_boards = all_moves_from_board(board, self.AI_SIGN)
			for new_board in new_boards:
				if game_won_by(new_board) == self.AI_SIGN:
					return new_board
			safe_moves = []
			for new_board in new_boards:
				if not player_can_win(new_board, self.OPPONENT_SIGN):
					safe_moves.append(new_board)
			return choice(safe_moves) if len(safe_moves) > 0 else \
				new_boards[0]

		def all_moves_from_board(board, sign):
			if sign == self.AI_SIGN:
				empty_field_count = board.count(self.EMPTY_SIGN)
				if empty_field_count == 9:
					return [sign + self.EMPTY_SIGN * 8]
				elif empty_field_count == 7:
					return [
						board[:8] + sign if board[8] == \
											self.EMPTY_SIGN else
						board[:4] + sign + board[5:]
					]
			move_list = []
			for i, v in enumerate(board):
				if v == self.EMPTY_SIGN:
					new_board = board[:i] + sign + board[i + 1:]
					move_list.append(new_board)
					if game_won_by(new_board) == self.AI_SIGN:
						return [new_board]
			if sign == self.AI_SIGN:
				safe_moves = []
				for move in move_list:
					if not player_can_win(move, self.OPPONENT_SIGN):
						safe_moves.append(move)
				return safe_moves if len(safe_moves) > 0 else \
					move_list[0:1]
			else:
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
