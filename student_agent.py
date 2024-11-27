# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves, get_directions



@register_agent("late_break_agent")
class LateBreakAgent(Agent):
  """
  A class for your implementation. Feel free to use this class to
  add any helper functionalities needed for your agent.
  """

  def __init__(self):
    super(LateBreakAgent, self).__init__()
    self.name = "StudentAgent"  


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
    time_taken = time.time() - start_time 
    time_limit = 2.00 #much lower than 2.0 as moves at a certain depth get quite slow
    best_move = None
    depth = 1
    board_size = chess_board.shape[0]
    value_matrix = self.get_value_map(chess_board, board_size)
    
    
    while time_taken < time_limit:
        valid_moves = get_valid_moves(chess_board, player)
        sorted_moves = self.sort_moves(valid_moves, value_matrix)
        
        for move in sorted_moves:
            new_board = deepcopy(chess_board)
            execute_move(new_board, move, player)
            elapsed_time = time.time() - start_time
            move_score = self.minimax(new_board, depth, float("-inf"), float("inf"), False, player, opponent, value_matrix, elapsed_time, time_limit)
            if best_move is None or move_score > best_move[1]:
                best_move = (move, move_score)
            # if time.time() - start_time >= 1.5: #if we estimate we don't have time to check another move, break out
            #     break

        depth += 1
        time_taken = time.time() - start_time

    print("My AI's turn took ", time_taken, "seconds.")

    if best_move:
        return best_move[0]
    else:
        return random_move(chess_board, player)

  def minimax(self, board, depth, alpha, beta, maximizing_player, player, opponent, value_matrix, elapsed_time : float, time_limit : float):
    start_time = time.time()
    if player == 1:
        is_endgame, player_score, opponent_score = check_endgame(board, player, opponent)
    else:
        is_endgame, opponent_score, player_score = check_endgame(board, opponent, player)
    if is_endgame:
        return player_score - opponent_score
    if depth == 0:
        return self.heuristic(board, player, opponent, len(value_matrix[0]))
    # if elapsed_time >= time_limit:
    #    return None

    valid_moves = get_valid_moves(board, player if maximizing_player else opponent)
    sorted_moves = self.sort_moves(valid_moves, value_matrix)
    
    if maximizing_player:
        max_eval = float("-inf")
        for move in sorted_moves:
            current_time = time.time() - start_time
            if current_time + elapsed_time >= time_limit:
               break
            new_board = deepcopy(board)
            execute_move(new_board, move, player)
            eval = self.minimax(new_board, depth - 1, alpha, beta, False, player, opponent, value_matrix, current_time + elapsed_time, time_limit)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break  #beta cutoff
        return max_eval
    else:
        min_eval = float("inf")
        for move in sorted_moves:
            current_time = time.time() - start_time
            if current_time + elapsed_time >= time_limit:
                break
            new_board = deepcopy(board)
            execute_move(new_board, move, opponent)
            eval = self.minimax(new_board, depth - 1, alpha, beta, True, player, opponent, value_matrix, current_time + elapsed_time, time_limit)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break  #alpha cutoff
        return min_eval

  def heuristic(self, board, player, opponent, board_size):
    game_stage = self.get_game_stage(board, player, opponent, board_size)

    #map control based off value map
    value_map = self.get_value_map(board, board_size)

    player_weighted_score = 0 #calculate the weighted score for the player and opponent
    opponent_weighted_score = 0

    for i in range(board_size):
        for j in range(board_size):
            if board[i, j] == player:
                player_weighted_score += value_map[i, j]
            elif board[i, j] == opponent:
                opponent_weighted_score += value_map[i, j]
    
    map_score = (player_weighted_score - opponent_weighted_score) / (player_weighted_score + opponent_weighted_score +1) #normalized

    #corner control: only if corners are taken. Take from gpt_greedy_corners_agent
    corners = [(0, 0), (0, board.shape[1] - 1), (board.shape[0] - 1, 0), (board.shape[0] - 1, board.shape[1] - 1)]
    corner_score = sum(1 for corner in corners if board[corner] == player) * 25
    corner_penalty = sum(1 for corner in corners if board[corner] == opponent) * -25

    #piece parity
    if player == 1:
        is_endgame, player_score, opponent_score = check_endgame(board, player, opponent)
    else:
        is_endgame, opponent_score, player_score = check_endgame(board, opponent, player)
    parity = (player_score - opponent_score) / (player_score + opponent_score +1)
    
    #mobility
    # heuristic 3: potential mobility
    # we can give an estimate for potential mobility by counting empty squares next to opposing pieces
    frontier_count = 0
    directions = get_directions()
    for i, j in np.ndindex(board.shape):
      if board[i, j] == opponent:
          for dx, dy in directions:
              ni, nj = i + dx, j + dy  
              # check for out of bounds and empty
              if 0 <= ni < board_size and 0 <= nj < board_size and board[ni, nj] == 0:
                  frontier_count += 1
    potential_mobility = frontier_count

    # opponent mobility
    opp_mobility = -len(get_valid_moves(board, opponent))

    #if game_stage == "early":
    #  return some weighting of factors: higher value to corners
    #elif game_stage == "mid":
    #  return some weighting of factors: higher value to stability and mobility
    #else:
    #  higher value to parity and stability
    
    return parity + 100 * map_score + 5*potential_mobility + corner_score + corner_penalty + opp_mobility


  def get_game_stage(self, board, player, opponent, board_size):
    """
    Define game state based off of the total number of stones on the board
    """
    if player == 1:
        is_endgame, player_score, opponent_score = check_endgame(board, player, opponent)
    else:
        is_endgame, opponent_score, player_score = check_endgame(board, opponent, player)
    prop_filled = (player_score + opponent_score / board_size * board_size) *100
    if prop_filled < 25:
      return "early"
    elif prop_filled < 85:
      return "mid"
    else:
      return "end"

  def sort_moves(self, valid_moves, value_matrix):
    sorted_moves = sorted( #sort moves according to value map
        valid_moves,
        key=lambda move: value_matrix[move[0], move[1]],  #access value_matrix[row, col]
        reverse=True
        )
    return sorted_moves

  def get_value_map(self, board, board_size):
    maps = { #weighted maps depending on board size, values computed wrt capturing a corner.
    6: np.array([
        [150, -75, 50, 50, -75, 150],
        [-75, -150, -25, -25, -150, -75],
        [50, -25, 0, 0, -25, 50],
        [50, -25, 0, 0, -25, 50],
        [-75, -150, -25, -25, -150, -75],
        [150, -75, 50, 50, -75, 150],]), 
    8: np.array([
        [200, -100, 100, 50, 50, 100, -100, 200],
        [-100, -200, -50, -50, -50, -50, -200, -100],
        [100, -50, 100, 0, 0, 100, -50, 100],
        [50, -50, 0, 0, 0, 0, -50, 50],
        [50, -50, 0, 0, 0, 0, -50, 50],
        [100, -50, 100, 0, 0, 100, -50, 100],
        [-100, -200, -50, -50, -50, -50, -200, -100],
        [200, -100, 100, 50, 50, 100, -100, 200],]),
    10: np.array([
        [250, -125, 100, 75, 50, 50, 75, 100, -125, 250],
        [-125, -250, -75, -50, -50, -50, -50, -75, -250, -125],
        [100, -75, 50, 25, 0, 0, 25, 50, -75, 100],
        [75, -50, 25, 0, 0, 0, 0, 25, -50, 75],
        [50, -50, 0, 0, 0, 0, 0, 0, -50, 50],
        [50, -50, 0, 0, 0, 0, 0, 0, -50, 50],
        [75, -50, 25, 0, 0, 0, 0, 25, -50, 75],
        [100, -75, 50, 25, 0, 0, 25, 50, -75, 100],
        [-125, -250, -75, -50, -50, -50, -50, -75, -250, -125],
        [250, -125, 100, 75, 50, 50, 75, 100, -125, 250],]),
    12: np.array([
        [300, -150, 125, 100, 75, 50, 50, 75, 100, 125, -150, 300],
        [-150, -300, -100, -75, -50, -50, -50, -50, -75, -100, -300, -150],
        [125, -100, 75, 50, 25, 0, 0, 25, 50, 75, -100, 125],
        [100, -75, 50, 25, 0, 0, 0, 0, 25, 50, -75, 100],
        [75, -50, 25, 0, 0, 0, 0, 0, 0, 25, -50, 75],
        [50, -50, 0, 0, 0, 0, 0, 0, 0, 0, -50, 50],
        [50, -50, 0, 0, 0, 0, 0, 0, 0, 0, -50, 50],
        [75, -50, 25, 0, 0, 0, 0, 0, 0, 25, -50, 75],
        [100, -75, 50, 25, 0, 0, 0, 0, 25, 50, -75, 100],
        [125, -100, 75, 50, 25, 0, 0, 25, 50, 75, -100, 125],
        [-150, -300, -100, -75, -50, -50, -50, -50, -75, -100, -300, -150],
        [300, -150, 125, 100, 75, 50, 50, 75, 100, 125, -150, 300],])}

    value_map = maps[board_size]

    if board[0, 0] != 0:
        value_map[:int(board_size/2), :int(board_size/2)] = 0  #tile values only relevant towards capturing corners. Change to 0 when a corner is captured

    if board[0, board_size - 1] != 0:
        value_map[:int(board_size/2), int(board_size/2):] = 0  # Zero out top-right 3x4 section
        #implement stable frontier: for each stable disc, update values around to 150. Worry about last.

    if board[board_size - 1, 0] != 0:
        value_map[board_size - int(board_size/2):, :int(board_size/2)] = 0  # Zero out bottom-left 3x4 section

    if board[board_size - 1, board_size - 1] != 0:
        value_map[board_size - int(board_size/2):, int(board_size/2):] = 0  # Zero out bottom-right 3x4 section
    
    return value_map

  # def stable_frontier(self, board, corner_c, corner_r, value_map, board_size, player, opponent):
  #   stable_discs = [(corner_r, corner_c)]
  #   directions = []
  #   if corner_c == 0:
  #     directions.append((1,0))
  #   elif corner_c == board_size - 1:
  #     directions.append((-1,0))
    
  #   if corner_r == 0:
  #     directions.append((0,-1))
  #   elif corner_r == board_size - 1:
  #     directions.append((1,0))

  #   for disc in stable_discs: #REVISIT : EDGE ADJ TO CORNER ALWAYS STABLE, DIAGONAL NOT STABLE UNLESS EDGE IS STABLE
  #     value_map[disc[0],disc[1]] = 150
  #     for dir in directions:
  #       value_map[disc[0]+dir[0], disc[1]+dir[1]] = 150 #stable disc: worth capturing regardless of who has corner
  #       if board[disc[0]+dir[0], disc[1]+dir[1]] == board[disc[0], disc[1]]: #if it's been captured by whoever holds the corresponding corner
  #         stable_discs.append(board[disc[0]+dir[0], disc[1]+dir[1]]) #add to stable
