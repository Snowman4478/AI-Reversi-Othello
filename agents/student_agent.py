# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves


class Node(object):
  board_state = None
  heuristicValue = 0
  montecarloSuccessAndTot = 0
  max = 1
  children = []
  move= (-1,-1)

  def __init__ (self, board_state, heuristicValue,montecarloSuccessAndTot, max, children, move):
    self.board_state = board_state
    self.heuristicValue= heuristicValue
    self.montecarloSuccessAndTot=montecarloSuccessAndTot
    self.max= max
    self.children = children
    self.move=move

def createNode(board_state, heuristicValue,montecarloSuccessAndTot, max, children, move):
  node = Node(board_state, heuristicValue,montecarloSuccessAndTot, max, children, move)
  return node



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
    next_move = self.treeStructure(chess_board, player, opponent, numberOFSimulations=20)
    time_taken = time.time() - start_time

    print("My AI's turn took ", time_taken, "seconds.")

    return next_move
  

  #see if next move you can grab a corner
  def taken_corner(self, chess_board, player):
    board_size = (chess_board.shape)[0]

    corners = [(0,0),(0,board_size-1),(board_size-1,0),(board_size-1,board_size-1)]
    corner_count = 0

    valid_moves = get_valid_moves(chess_board, player)

    if set(corners) & set(valid_moves):
      return 100
    return 0

  
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

      #corner peice
      if (((r==0) or (r==board_size-1)) and ((c==0) or (c==board_size-1))):
        fixed_count += 1
        continue
      #if its on the sides just check if that side is full, only way it can be flipped
      elif (r == 0) or (r == board_size - 1):
        if row_check:
          fixed_count += 1
          continue
      elif (c == 0) or (c == board_size - 1):
        if col_check:
          fixed_count += 1
          continue
      else:
        directions = [(-1,-1),(-1,1),(1,-1),(1,1)]
        for dx,dy in directions:
          rdiag = r
          cdiag = c
          diag_check = True
          while ((0 <= rdiag <= board_size-1) and (0 <= cdiag <= board_size-1)):
            if chess_board[rdiag][cdiag] == 0:
              #if there is a 0 in the diag then piece (roughly) can't be fixed
              diag_check = False
              break
            rdiag += dx
            cdiag += dy
          #break out early to save time computing if 0 found
          if diag_check == False:
            break

        if row_check and col_check and diag_check:
            fixed_count += 1
        

    return 100*fixed_count/len(player_pieces_indices)
  

  #calculates adjacent empty sqaures of player - opponent and returns normalized value
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
      
    return 100*(len(player_seen) - len(opp_seen))/(len(opp_seen) + len(player_seen)+1)
  

  #calculates players pieces - opponents and returns normalized value
  def piece_difference(self, chess_board, player, opponent):

    num_of_player_pieces = len(np.argwhere(chess_board == player))
    num_of_opponent_pieces = len(np.argwhere(chess_board == opponent))

    return 100*(num_of_player_pieces - num_of_opponent_pieces)/(1+num_of_player_pieces + num_of_opponent_pieces )


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
        return 100
    
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
    return 100*(abs(player_score) - abs(opponent_score)) / (1 + abs(player_score) + abs(opponent_score))
  

  #calculate the number of moves you can make vs opponent
  def available_moves(self, chess_board,player,opponent):
    player_moves = len(get_valid_moves(chess_board, player))
    opponent_moves = len(get_valid_moves(chess_board, opponent))

    return 100*(player_moves - opponent_moves)/(player_moves + opponent_moves + 1)
  

  #see how many full lines on the side we have
  def full_side(self, chess_board):
    board_size = (chess_board.shape)[0]

    row0_check = np.all(chess_board[0, :] != 0)
    col0_check = np.all(chess_board[:, 0] != 0)
    rown_check = np.all(chess_board[board_size-1, :] != 0)
    coln_check = np.all(chess_board[:, board_size-1] != 0)

    sum_of_sides = row0_check + col0_check + rown_check + coln_check

    return 100*sum_of_sides/4


  #our actual heuristic function will be a linear combination of the previous factors
  def heuristic_function(self, chess_board, move, player, opponent):
    chess_board_copy = deepcopy(chess_board)

    execute_move(chess_board_copy, move, player)
    #weights will be in the following order:
    #taken_corner, fixed_pieces, adjacent_spaces, piece_diff
    #does_opponent_pass, placement_score, available_moves, full_side 
    weights = [0, 0, 0, 0, 0, 0, 0, 0]

    nonzero_pieces = np.count_nonzero(chess_board_copy)
    total_pieces = np.size(chess_board_copy)

    #we calculate how far along we are in the game by looking at the pieces placed
    cur_progress = nonzero_pieces/total_pieces

    #if we are near the start of the game, weight differently
    if cur_progress <= 1/4:
      weights = [0.9, 0.6, 0.4, 0, 0, 0.4, 0.3, 0.6]
    #midgame
    elif 1/4 < cur_progress and cur_progress <= 2/3:
      weights = [0.9, 0.6, 0.4, 0.2, 0.1, 0.4, 0.2, 0.6]
    #endgame
    else:
      weights = [0.9, 0.6, 0.4, 0.4, 0.6, 0.4, 0, 0.6]
    
    #if weight = 0 for a process we can save time by just not computing it,
    #only weights that can be 0 are piece_diff, opp_pass and available moves
    corner_heuristic = self.taken_corner(chess_board_copy, player)
    
    fixed_heuristic = self.fixed_peices(chess_board_copy, player)

    adjacent_heuristic = self.adjacent_spaces(chess_board_copy, player, opponent)

    if weights[3] == 0:
      diff_heuristic = 0
    else:
      diff_heuristic = self.piece_difference(chess_board_copy, player, opponent)
    
    if weights[4] == 0:
      pass_heuristic = 0
    else:
      pass_heuristic = self.does_opponent_pass(chess_board_copy, player, opponent)
    
    score_heuristic = self.placement_score(chess_board_copy, player, opponent)

    if weights[6] == 0:
      avail_heuristic = 0
    else:
      avail_heuristic = self.available_moves(chess_board_copy, player, opponent)
    
    side_heuristic = self.full_side(chess_board_copy)
    
    heuristic_variables = [corner_heuristic, fixed_heuristic, adjacent_heuristic, 
                           diff_heuristic, pass_heuristic, 
                           score_heuristic, avail_heuristic, side_heuristic]
    
    #get weighted values
    weighted_values = [var * weight for var, weight in zip(heuristic_variables, weights)]

    return sum(weighted_values)


  #monteCarlo function,FASTER, returns the average heuristic value of the board after certain number of steps
  def monteCarloFaster(self, chess_board, totalSim, player, opponent, steps):
    total=totalSim
    heuristics=0
    while (totalSim >=0):
      stepCopy=steps 
      stepCopy=2*stepCopy
      chess_board_copy = deepcopy(chess_board)
      i=0
      start = time.time()
      while (stepCopy>0) :
        if (i%2 == 0): #player's move
          if(get_valid_moves(chess_board_copy, player) != []):
            execute_move(chess_board_copy, random_move(chess_board_copy, player), player)
        else: #opponent's move
          if(get_valid_moves(chess_board_copy, opponent) != []):
            execute_move(chess_board_copy, random_move(chess_board_copy, opponent), opponent)
        i=i+1
        stepCopy=stepCopy-1
      #no moves left, the end of game
      if(get_valid_moves(chess_board_copy, player) == []):
          if(get_valid_moves(chess_board_copy, opponent) == []):
            is_end,bluePlayerSc, brownPlayerSc = check_endgame(chess_board_copy, player , opponent)
            if (player == 1 and bluePlayerSc > brownPlayerSc) or (player == 2 and brownPlayerSc > bluePlayerSc):
              heuristic_value= 100
            else: 
              heuristic_value= -100
          else:
            heuristic_value = -100
      else:
        heuristic_value = self.heuristic_function(chess_board_copy,random_move(chess_board_copy, player),player, opponent)
      totalSim=totalSim-1
      heuristics= heuristics+heuristic_value
      
    return heuristics/total #returns average


  #Creating Nodes and links between them for pruning.
  def treeStructure(self, chess_board , player ,opponent, numberOFSimulations):
    ###check the board size for simulation adjustments
    board_size = (chess_board.shape)[0]
    steps = 0
    if(board_size == 6 ):
      numberOFSimulations = 100
      steps = 4

    if(board_size == 8 ):
      numberOFSimulations = 60
      steps = 4

    if(board_size == 10):
      numberOFSimulations = 40
      steps = 4
    
    if(board_size ==12):
      numberOFSimulations = 35
      steps = 4
    
    chess_board_copy= deepcopy(chess_board)

    grandParent= createNode(chess_board_copy, 0, 0 , max = 1 , children = list(), move=(-1,-1)) #max node
    GPmoves = get_valid_moves(chess_board_copy, player) #players valid moves

    #if corner is the move, instantly take
    board_size = (chess_board_copy.shape)[0]
    corners = set([(0,0),(0,board_size-1),(board_size-1,0),(board_size-1,board_size-1)]) 

    #sort grandParent moves by heuristic values descending also check if a corner
    #is apart of the moves and if so take the best corner.
    heuristics = []
    our_corners= []
    for GPmove in GPmoves:
      var = self.heuristic_function(chess_board_copy, GPmove, player, opponent)
      if GPmove in corners:
        our_corners.append((GPmove, var))
      heuristics.append((GPmove,var))
    heuristics.sort( key=lambda tup: tup[1], reverse=True)
    our_corners.sort( key=lambda tup: tup[1], reverse=True)
    if our_corners:
      return (our_corners[0])[0]
    
    i=0
    #created all children of all parents and put them in a list of the grandParent with respect to their heuristic values.
    for GPMove,GPHvalue in heuristics : 
      if (i < 6):
        parentsBoard = deepcopy(chess_board_copy)
        execute_move(parentsBoard, GPMove , player) # parents board

        parent= createNode(parentsBoard, GPHvalue, 0  , max=0, children = list() , move = GPMove ) #min node with no children
        averageForParent= self.monteCarloFaster(parentsBoard, numberOFSimulations, player, opponent, steps)
        parent.montecarloSuccessAndTot = averageForParent

        grandParent.children.append(parent)
        i=i+1
    
    max = -np.inf
    childTemp = Node(None, 0,0,1 ,list(),(-1,-1))
    for child in grandParent.children:
      if child.montecarloSuccessAndTot > max :
        max= child.montecarloSuccessAndTot
        childTemp=child 
    
    myset=set(GPmoves)
    if childTemp.move in myset : 
      return childTemp.move # normal algorithm return
    else: 
      return grandParent.children[0] #returns heuristicly better move if program chose an invalid move














