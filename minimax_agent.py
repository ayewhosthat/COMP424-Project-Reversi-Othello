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
  
  def MiniMaxValue(self, board_state, turn, player, opponent):
    # turn = 0 means that it's min player turn
    # turn = 1 means that it's max player turn
    game_over, p_score, o_score = check_endgame(board_state, player, opponent)
    if game_over:
      return self.evaluate_game_board(board_state, player, opponent, p_score, o_score)
      # evaluate the board if the game is deemed over

    moves = get_valid_moves(board_state, turn)
    utilities = [None]*len(moves)
    for j, move in enumerate(moves):
      copy = deepcopy(board_state)
      execute_move(copy, move, turn)
      utilities[j] = self.MiniMaxValue(copy, turn^1, player, opponent)
    return max(utilities) if turn else min(utilities)
  
  def MiniMaxDecision(self, board, player):
    moves = get_valid_moves(board, player)
    if not moves:
      return None
    vals = [None]*len(moves)
    for i, move in enumerate(moves):
      copy = deepcopy(board)
      execute_move(copy, move, player)
      vals[i] = (move, self.MiniMaxValue(copy, 0, player, 3-player))
    vals.sort(key=lambda game: game[1], reverse=False)
    return vals[0][0]

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
    best_move = self.MiniMaxDecision(chess_board, player)
    time_taken = time.time() - start_time

    print("My AI's turn took ", time_taken, "seconds.")

    # Dummy return (you should replace this with your actual logic)
    # Returning a random valid move as an example
    return best_move
    
