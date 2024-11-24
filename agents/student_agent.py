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
  

  #counts corners owned by player
  def taken_corner(self, chess_board, player):
    board_size = (chess_board.shape)[0]

    corners = [(0,0),(0,board_size-1),(board_size-1,0),(board_size-1,board_size-1)]
    corner_count = 0

    for corner_x,corner_y in corners:
      if chess_board[corner_x,corner_y] == player:
        corner_count +=1

    return corner_count/4

  
  #calculates how many pieces are unable to be moved
  def fixed_peices(self, chess_board, player):
    board_size = (chess_board.shape)[0]
    #pieces that are "locked in" froma all sides, can't be flipped thus we check
    #the sides and diagonals to see if they are filled in
    player_pieces_indices = np.argwhere(chess_board == player)
    player_pieces = [tuple(indx) for indx in player_pieces_indices]

    fixed_count = 0.0

    for piece in player_pieces:
      r, c = piece
      row_check = np.all(chess_board[r, :] != 0)
      col_check = np.all(chess_board[:, c] != 0)

      #if its on the sides just check if that side is full, only way it can be flipped
      if (r == 0) or (r == board_size - 1):
        if row_check:
          fixed_count += 1
          continue
      elif (c == 0) or (c == board_size - 1):
        if col_check:
          fixed_count += 1
          continue
      else:
        #TODO
        #checking the diagonals can be extremely time consuming, will have to speak about this
        #later
        continue

    return fixed_count/len(player_pieces_indices)
  

  #calculates adjacent empty sqaures of opponent - player and returns normalized value
  def adjacent_spaces(self, chess_board, player, opponent):
    board_size = chess_board.shape[0]
    
    player_pieces_indices = np.argwhere(chess_board == player)
    #argwhere doesn't return tuples :(
    player_pieces = [tuple(indx) for indx in player_pieces_indices]

    opponent_pieces_indices = np.argwhere(chess_board == opponent)
    opponent_pieces = [tuple(indx) for indx in opponent_pieces_indices]


    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]


    player_seen = set()
    opp_seen = set()
    
    
    for dir in directions:
      dx, dy = dir

      for (r,c) in player_pieces:
        r += dx
        c += dy
        #check to see if square is outside chess board
        if not (0 <= r < board_size and 0 <= c < board_size):
          continue

        if chess_board[r,c] == 0:
          player_seen.add((r,c))
      
      for (r,c) in opponent_pieces:
        r += dx
        c += dy
        #check to see if square is outside chess board
        if not (0 <= r < board_size and 0 <= c < board_size):
          continue

        if chess_board[r,c] == 0:
          opp_seen.add((r,c))
      
    return (len(opp_seen) - len(player_seen))/(len(opp_seen) + len(player_seen))
  

  #calculates players pieces - opponentes and returns normalized value
  def piece_difference(self, chess_board, player, opponent):

    num_of_player_pieces = len(np.argwhere(chess_board == player))
    num_of_opponent_pieces = len(np.argwhere(chess_board == opponent))

    return (num_of_player_pieces - num_of_opponent_pieces)/(num_of_player_pieces + num_of_opponent_pieces)


  #calculates if the next move causes opponent to not be able to play
  def does_opponent_pass(self, chess_board, player, opponent):
    player_moves = get_valid_moves(chess_board, player)
    opponent_moves = get_valid_moves(chess_board, opponent)
      
    player_score = np.sum(chess_board == player)
    opponent_score = np.sum(chess_board == opponent)

    #we want to see if this board makes it so the opponent can't go,however if 
    #player also can't go and player has less score, that would be an auto loss
    if not opponent_moves:
      if player_moves or (player_score > opponent_score):
        return 1.0
    
    return 0.0
  

  #calculates the score of the current placed pieces using our scoring system
  #this is a helper function so we can normalize it
  def placement_score_calc(self, chess_board, cur_player):
    #we will do a scale of -10 to 10 where -10 is the worst 
    #position and 10 is the best.
    board_size = (chess_board.shape)[0]

    player_pieces_indices = np.argwhere(chess_board == cur_player)
    player_pieces = [tuple(indx) for indx in player_pieces_indices]
    player_piece_score = 0
    sum = 0

    corners = set([(0,0),(0,board_size-1),(board_size-1,0),(board_size-1,board_size-1)])

    for piece in player_pieces:
      r, c = piece

      #10 score if peice in a corner
      if piece in corners:
        player_piece_score += 10
        continue

      #if piece is in a position that gives opponent the corner its bad,unless 
      #corner is already taken

      #top left (0,0)
      if piece in [(0,1),(1,0),(1,1)]:
        #if corner is free
        if chess_board[0,0] == 0:
          if piece in [(0,1),(1,0)]:
            #if adjacent to corner give a -5
            player_piece_score -= 5
            continue
          else:
            #if directly diagonal to corner give a -10 because its a really bad move
            player_piece_score -= 10
            continue
        else:
          continue
        
      #top right (0, board_size - 1)
      if piece in [(0, board_size-2), (1, board_size-1), (1, board_size-2)]:
        if chess_board[0, board_size- 1] == 0:
            if piece in [(0, board_size-2), (1, board_size-1)]:
                player_piece_score -= 5
                continue
            else:
                player_piece_score -= 10
                continue
        else:
          continue
      
      #bottom left (board_size-1, 0)
      if piece in [(board_size-2, 0), (board_size-1, 1), (board_size-2, 1)]:
          if chess_board[board_size-1, 0] == 0:
              if piece in [(board_size-2, 0), (board_size-1, 1)]:
                  player_piece_score -= 5
                  continue
              else:
                  player_piece_score -= 10
                  continue
          else:
            continue
      
      #bottom right corner (board_size- 1, board_size-1)
      if piece in [(board_size-2, board_size-1), (board_size-1, board_size-2), (board_size-2, board_size-2)]:
          # If the corner is free
          if chess_board[board_size-1, board_size-1] == 0:
              if piece in [(board_size-2, board_size-1), (board_size-1, board_size-2)]:
                  player_piece_score -= 5
                  continue
              else:
                  player_piece_score -= 10
                  continue
          else:
            continue
      
      #if piece is on the sides its favourable thus +5
      if ((r==0) or (r==board_size-1) or (c==0) or (c==board_size-1)):
        player_piece_score += 5
        continue

      #if piece is right before a side then its a bad spot but not too bad so -2.5
      if ((r==1) or (r==board_size-2) or (c==1) or (c==board_size-2)):
        player_piece_score -= 2.5
        continue

      #lastly if piece is right before the really bad diagonal (the -10)
      #then that a good piece to have cause it leads to opponent having the bad spot
      if piece in [(2,2),(2,board_size-3),(board_size-3,2),(board_size-3,board_size-3)]:
        player_piece_score += 5

      #any other tile is counted as 0 as it's not too relevant
      return player_piece_score
    
  
  #this is the placement_score function we will actually use for the heuristic
  def placement_score(self, chess_board, player, opponent):
    player_score = self.placement_score_calc(chess_board, player)
    opponent_score = self.placement_score_calc(chess_board, opponent)

    #add 1 to the denominator cause at beginning of the game potential to have 0 score in tiles
    return (player_score - opponent_score) / (1 + player_score + opponent_score)
  

  #calculate the number of moves you can make vs opponent
  def available_moves(self, chess_board,player,opponent):
    player_moves = len(get_valid_moves(chess_board, player))
    opponent_moves = len(get_valid_moves(chess_board, opponent))

    return (player_moves - opponent_moves)/(player_moves + opponent_moves + 1)


  #our actual heuristic function will be a linear combination of the previous factors
  def heuristic_function(self, chess_board, player, opponent):
    #weights will be in the following order:
    #taken_corner, fixed_pieces, adjacent_spaces, piece_diff
    #does_opponent_pass, placement_score, available_moves 
    weights = [0, 0, 0, 0, 0, 0, 0]

    nonzero_pieces = np.count_nonzero(chess_board)
    total_pieces = np.size(chess_board)

    #we calculate how far along we are in the game by looking at the pieces placed
    cur_progress = nonzero_pieces/total_pieces

    #if we are near the start of the game, weight differently
    if cur_progress <= 1/3:
      #3 is highest weight, 1 is lowest, 0 is no weight
      weights = [3, 3, 2, 0, 0, 2, 3]
    #midgame
    elif 1/3 < cur_progress and cur_progress <= 2/3:
      weights = [3, 3, 2, 1, 2, 2, 1]
    #endgame
    else:
      weights = [3, 3, 2, 2, 3, 2, 0]
    
    #if weight = 0 for a process we can save time by just not computing it,
    #only weights that can be 0 are piece_diff, opp_pass and available moves
    corner_heuristic = self.taken_corner(chess_board, player)
    
    fixed_heuristic = self.fixed_peices(chess_board, player)

    adjacent_heuristic = self.adjacent_spaces(chess_board, player, opponent)

    if weights[3] == 0:
      diff_heuristic = 0
    else:
      diff_heuristic = self.piece_difference(chess_board, player, opponent)
    
    if weights[4] == 0:
      pass_heuristic = 0
    else:
      pass_heuristic = self.does_opponent_pass(chess_board, player, opponent)
    
    score_heuristic = self.placement_score(chess_board, player, opponent)

    if weights[6] == 0:
      avail_heuristic = 0
    else:
      avail_heuristic = self.available_moves(chess_board, player, opponent)
    
    heuristic_variables = [corner_heuristic, fixed_heuristic, adjacent_heuristic, 
                           diff_heuristic, pass_heuristic, score_heuristic, avail_heuristic]
    
    #get weighted values
    weighted_values = [var * weight for var, weight in zip(heuristic_variables, weights)]

    return sum(weighted_values)


















