# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves

@register_agent("minimax_agent")
class MinimaxAgent(Agent):
  """
  A class for your implementation. Feel free to use this class to
  add any helper functionalities needed for your agent.
  """

  def __init__(self):
    super(MinimaxAgent, self).__init__()
    self.name = "StudentAgent"

  def evaluate_game_board(self, board, colour, opponent_colour, player_score, opponent_score):
    # SUBJECT TO CHANGE

    # Metric 2: mobility (number of available moves for us compared to number of available moves for them)
    mobility = -len(get_valid_moves(board, opponent_colour))

    # Metric 3: piece parity
    # computed in the same iteration as for weighting to save time

    # Metric 4: corner control
    corners = [(0, 0), (0, board.shape[1] - 1), (board.shape[0] - 1, 0), (board.shape[0] - 1, board.shape[1] - 1)]
    corner_score = sum(1 for corner in corners if board[corner] == colour)
    corner_penalty = sum(1 for corner in corners if board[corner] == 3 - colour)
    return 25*(corner_score - corner_penalty) + mobility + (player_score - opponent_score)
  
  def MiniMaxDecision(self, board_state, player, opponent):
    moves = get_valid_moves(board_state, player)
    if not moves:
      return None
    highest_value = float('-inf')
    best_move = None
    for move in moves:
      new_state = deepcopy(board_state)
      execute_move(new_state, move, player)
      value = self.MiniMaxValue(new_state, player, opponent, opponent)
      if value > highest_value:
        best_move = move
        highest_value = value
    return best_move

  def MiniMaxValue(self, state, player, opponent, to_move):
    game_over = None
    p_score = None
    o_score = None
    check = None
    if to_move == player:
      # our turn to move
      check = check_endgame(state, player, opponent)
    elif to_move == opponent:
      # opponent's turn to move
      check = check_endgame(state, opponent, player)
    game_over = check[0]

    if player == 1:
        p_score = check[1]
        o_score = check[2]
    elif player == 2:
        p_score = check[2]
        o_score = check[1]
    if game_over:
      # terminal state
      return self.evaluate_game_board(state, player, opponent, p_score, o_score)
    
    moves = get_valid_moves(state, to_move)
    if not moves:
      return self.MiniMaxValue(state, player, opponent, 3 - to_move)

    if to_move == player:
      max_val = float('-inf')
      for move in moves:
        new_board = deepcopy(state)
        execute_move(new_board, move, to_move)
        val = self.MiniMaxValue(new_board, player, opponent, 3 - to_move)
        if val > max_val:
          max_val = val
      return max_val
    else:
      min_val = float('inf')
      for move in moves:
        new_board = deepcopy(state)
        execute_move(new_board, move, to_move)
        val = self.MiniMaxValue(new_board, player, opponent, 3 - to_move)
        if val < min_val:
          min_val = val
      return min_val

  def step(self, chess_board, player, opponent):
    """
    Implement the step function of your agent here.
    You can use the following variables to access the chess board:
    - chess_board: a numpy array of shape (board_size, board_size)
      where 0 represents an empty spot, 1 represents Player 1's discs (Blue),
      and 2 represents Player 2's discs (Brown).
    - player: 1 if this agent is playing as Player 1 (Blue), or 2 if playing as Player 2 (Brown).
    - opponent: 1 if the opponent is Player 1 (Blue), or 2 if the opponent is Player 2 (Brown).

    You should return a tuple (r,c), where (r,c) is the position where your agent
    wants to place the next disc. Use functions in helpers to determine valid moves
    and more helpful tools.

    Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
    """

    # Some simple code to help you with timing. Consider checking 
    # time_taken during your search and breaking with the best answer
    # so far when it nears 2 seconds.
    start_time = time.time()
    best = self.MiniMaxDecision(chess_board, player, opponent)
    time_taken = time.time() - start_time

    print("My AI's turn took ", time_taken, "seconds.")

    # Dummy return (you should replace this with your actual logic)
    # Returning a random valid move as an example
    
    return best
