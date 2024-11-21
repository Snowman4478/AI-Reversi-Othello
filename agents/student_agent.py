# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves

@register_agent("student_agent")
class StudentAgent(Agent):
  """
  A class for your implementation. Feel free to use this class to
  add any helper functionalities needed for your agent.
  """

  def __init__(self):
    super(StudentAgent, self).__init__()
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

    print("My AI's turn took ", time_taken, "seconds.")

    # Dummy return (you should replace this with your actual logic)
    # Returning a random valid move as an example
    return random_move(chess_board,player)
  
  def is_corner_available(chess_board, valid_moves):
    board_size = (chess_board.shape)[0]

    corners = [(0,0),(0,board_size-1),(board_size-1,0),(board_size-1,board_size-1)]

    #turn into a set so you can & them to see if theres a match in both lists
    if bool(set(corners) & set(valid_moves)):
      #return double instead of bool for weighting
      return 1.0
    else:
      return 0.0
  
  def fixed_peices(chess_board):
    ##Tough gotta think about this some more
    return 1.0
  
  def adjacent_spaces(chess_board, player, oppenent):
    board_size = chess_board.shape[0]
    
    player_pieces_indices = np.argwhere(chess_board == player)
    #argwhere doesn't return tuples :(
    player_pieces = [tuple(indx) for indx in player_pieces_indices]

    opponent_pieces_indices = np.argwhere(chess_board == oppenent)
    opponent_pieces = [tuple(indx) for indx in opponent_pieces_indices]

    opp_adjacent = 0
    player_adjacent = 0 

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    
    for dir in directions:
      dx, dy = dir

      for (r,c) in player_pieces:
        r += dx
        c += dy
        #check to see if square is outside chess board
        if not (0 <= r < board_size and 0 <= c < board_size):
          continue

        if chess_board[r,c] == 0:
          player_adjacent += 1
      
      for (r,c) in opponent_pieces:
        r += dx
        c += dy
        #check to see if square is outside chess board
        if not (0 <= r < board_size and 0 <= c < board_size):
          continue

        if chess_board[r,c] == 0:
          opp_adjacent += 1
      
      return (opp_adjacent - player_adjacent)/(opp_adjacent + player_adjacent)




