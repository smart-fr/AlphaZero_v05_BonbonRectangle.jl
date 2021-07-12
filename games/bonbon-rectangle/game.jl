import AlphaZero.GI

using Crayons
using StaticArrays

const BOARD_SIDE = 16
const NUM_CELLS = Int(BOARD_SIDE) ^ 2
const TO_CONQUER = 0.875
const EXPAND_PERIOD = 20
const MAX_EXPANSIONS = 8
# Initial board as a list of bonbon notations
const INITIAL_BOARD_SIZE_16_LIST = [
  "00037", "0083B", "04073", "08CBF", "0C4F7", "0C8FF",
  "10C7F", "144BB", "180F3"
]

const Player = UInt8
const WHITE = 0x01
const BLACK = 0x02
other_player(p::Player) = 0x03 - p

const Cell = UInt8
const ZERO_BOARD = @SMatrix zeros(Cell, BOARD_SIDE, BOARD_SIDE)
const Board = typeof(ZERO_BOARD)
const EMPTY_TUPLE = @SMatrix [(-1, -1) for row in 1:BOARD_SIDE, column in 1:BOARD_SIDE]
const ZERO_TUPLE = @SMatrix [(0, 0) for row in 1:BOARD_SIDE, column in 1:BOARD_SIDE]
const BoardTuple = typeof(EMPTY_TUPLE)
const EMPTY_STATE = (board=ZERO_BOARD, impact=ZERO_TUPLE, actions_hook=EMPTY_TUPLE, curplayer=WHITE)

struct GameSpec <: GI.AbstractGameSpec end

mutable struct GameEnv <: GI.AbstractGameEnv
  board :: Board
  curplayer :: Player
  finished :: Bool
  winner :: Player
  # Actions mask where each flagged action is attached to a bonbon's NW corner
  amask :: Vector{Bool}
  amask_white :: Vector{Bool}
  amask_black :: Vector{Bool}
  # Cell indices of NW corner of bonbon where current cell belongs,
  # where all legal actions for the bonbon are flagged in GameEnv.amask
  actions_hook :: BoardTuple
  # Actions history, which uniquely identifies the current board position
  # Used by external solvers and to trigger expansions through turns count
  history :: Union{Nothing, Vector{Int}}
  impact :: BoardTuple
end

GI.spec(::GameEnv) = GameSpec()

###################################
# CELLS                           #
###################################
#
# Cell values are encoded as 0-127 integers with bits set according to:
# Bit 6: set if the cell belongs to a former attacking or defending bonbon
const CELL_FORMER_FIGHTER = 0x40
# Bit 5: set if cell's W side is a vertical limit 
const CELL_V_BORDER_WEST = 0x20
# Bit 4: set if cell's S side is a horizontal limit
const CELL_H_BORDER_SOUTH = 0x10
# Bit 3: set if cell's E side is a vertictal limit
const CELL_V_BORDER_EAST = 0x08
# Bit 2: set if cell's N side is a horizontal limit
const CELL_H_BORDER_NORTH = 0x04
# Bit 1: set if the cell is black
# Bit 0: set if the cell is white
# --------------------
# FF We So Ea No Bl Wh
#  4  2  1  8  4  2  1
# --------------------
#        1           1
#     2           2
#     3  3        3  3
#  4           4
#  5     5     5     5
#  6  6        6  6
#  7     7     7     7
#           8
#           9        9
#           A     A
#           B     B  B
#           C  C
#           D        D
#           E     E  E
#           F  F  F  F
# --------------------
#  4  2  1  8  4  2  1
# FF We So Ea No Bl Wh
# --------------------
#
###
# Convert between 1-based cell indices and 0-based (column, row) tuples.
###
# To iterate cells from left to right, top to bottom :
# cell_index = 0xRC+1 -> cell CR in bonbon notation.
# cell_index_to_column_row(cell_index::Int) = ((cell_index - 1) % BOARD_SIDE, (cell_index - 1) ÷ BOARD_SIDE)
# colum_row_to_cell_index((column, row)) = column + (row * BOARD_SIDE) + 1
# To iterate cells from top to bottom, left to right :
# cell_index = 0xCR+1 -> cell CR in bonbon notation.
index_to_column_row(cell_index::Int) = ((cell_index - 1) ÷ BOARD_SIDE, (cell_index - 1) % BOARD_SIDE)
column_row_to_index((column, row)) = row + (column * BOARD_SIDE) + 1
#
# Cell tests
#
# Test for cell properties from its value
cell_value_is_former_fighter(cell_value::Cell) = (cell_value & CELL_FORMER_FIGHTER) == CELL_FORMER_FIGHTER
cell_value_has_N_border(cell_value::Cell) = (cell_value & CELL_H_BORDER_NORTH) == CELL_H_BORDER_NORTH
cell_value_has_S_border(cell_value::Cell) = (cell_value & CELL_H_BORDER_SOUTH) == CELL_H_BORDER_SOUTH
cell_value_has_W_border(cell_value::Cell) = (cell_value & CELL_V_BORDER_WEST) == CELL_V_BORDER_WEST
cell_value_has_E_border(cell_value::Cell) = (cell_value & CELL_V_BORDER_EAST) == CELL_V_BORDER_EAST
cell_value_is_white(cell_value::Cell) = (cell_value & WHITE) == WHITE
cell_value_is_black(cell_value::Cell) = (cell_value & BLACK) == BLACK
cell_value_is_empty(cell_value::Cell) = !cell_value_is_white(cell_value) && !cell_value_is_black(cell_value)
#
# Test for board cell properties BloWhits coordinates
#
# Generic test: return a function accepting a board and coordinates as arguments, which
# returns the return of "test_function applied" to the corresponding cell value in the board.
test_board_column_row(test_function) = (board, column, row) -> (0 <= column < BOARD_SIDE) && (0 <= row < BOARD_SIDE) && test_function(board[row + 1, column + 1])
test_out_or_board_column_row(test_function) = (board, column, row) -> !(0 <= column < BOARD_SIDE) || !(0 <= row < BOARD_SIDE) || test_function(board[row + 1, column + 1])
#
# Applications
is_former_fighter = test_board_column_row(cell_value_is_former_fighter)
has_N_border = test_board_column_row(cell_value_has_N_border)
has_S_border = test_board_column_row(cell_value_has_S_border)
has_W_border = test_board_column_row(cell_value_has_W_border)
has_E_border = test_board_column_row(cell_value_has_E_border)
is_white = test_board_column_row(cell_value_is_white)
is_black = test_board_column_row(cell_value_is_black)
is_empty = test_out_or_board_column_row(cell_value_is_empty)
#
# Converts a boolean array to an integer by interpreting the array as its binary representation,
# from the least (1st array element) to the most (last array element) significant bits.
function booleans_to_integer(array::Array)
  if length(array) == 0
    return 0
  else
    return 2 * booleans_to_integer(array[2:end]) + (array[1] ? 1 : 0)
  end
end
# Converts a boolean array to a string by concatenating the hexadecimal representations of the
# return of booleans_to_integer for each group of 4 (or less for the last ones) elements.
function booleans_to_string(array::Array)
  try
    result = ""
    if 0 < length(array) <= 4
      return string(booleans_to_integer(reverse(array)), base=16)
    elseif length(array) > 4
      return string(booleans_to_integer(reverse(array[1:4])), base=16) * booleans_to_string(array[5:end])
    end
    return result
  catch
    return nothing
  end
end
#
# Convert a String to a boolean array by interpreting the string as the hexadecimal representation of bits
# representing the boolean values.
function string_to_booleans(s::String)
  try
    s == "" ? [] : map(!=('0'), [(map(x -> bitstring(x)[5:8], map(c -> parse(UInt8, c, base=16), collect(s)))...)...])
  catch
    return nothing
  end
end
#
# Cell modifiers
#
# Change properties to cell and reflect them in cell value
# Set bits
cell_value_set_former_fighter(cell_value::Cell) = cell_value_is_former_fighter(cell_value) ? cell_value : cell_value + CELL_FORMER_FIGHTER
cell_value_add_N_border(cell_value::Cell) = cell_value_has_N_border(cell_value) ? cell_value : cell_value + CELL_H_BORDER_NORTH
cell_value_add_S_border(cell_value::Cell) = cell_value_has_S_border(cell_value) ? cell_value : cell_value + CELL_H_BORDER_SOUTH
cell_value_add_W_border(cell_value::Cell) = cell_value_has_W_border(cell_value) ? cell_value : cell_value + CELL_V_BORDER_WEST
cell_value_add_E_border(cell_value::Cell) = cell_value_has_E_border(cell_value) ? cell_value : cell_value + CELL_V_BORDER_EAST
function cell_value_set_empty(cell_value::Cell)
  v = cell_value_is_black(cell_value) ? cell_value - BLACK : cell_value
  return cell_value_is_white(v) ? v - WHITE : v
end
function cell_value_set_white(cell_value::Cell)
  v = cell_value_is_black(cell_value) ? cell_value - BLACK : cell_value
  return cell_value_is_white(v) ? v : v + WHITE
end
function cell_value_set_black(cell_value::Cell)
  v = cell_value_is_white(cell_value) ? cell_value -  WHITE : cell_value
  return cell_value_is_black(v) ? v : v + BLACK
end
# Unset bits
cell_value_unset_former_fighter(cell_value::Cell) = cell_value_is_former_fighter(cell_value) ? cell_value - CELL_FORMER_FIGHTER : cell_value
cell_value_remove_N_border(cell_value::Cell) = cell_value_has_N_border(cell_value) ? cell_value - CELL_H_BORDER_NORTH : cell_value
cell_value_remove_S_border(cell_value::Cell) = cell_value_has_S_border(cell_value) ? cell_value - CELL_H_BORDER_SOUTH : cell_value
cell_value_remove_W_border(cell_value::Cell) = cell_value_has_W_border(cell_value) ? cell_value - CELL_V_BORDER_WEST : cell_value
cell_value_remove_E_border(cell_value::Cell) = cell_value_has_E_border(cell_value) ? cell_value - CELL_V_BORDER_EAST : cell_value
#
# Change properties to cell and update the board to reflect the changes
#
# Generic modifier: return a function accepting a board and coordinates as arguments,
# which updates the corresponding cell value in the board by application of "modify_function", except
# if coordinates are outside the board.
function set_board_column_row(modify_function)
  return function(board::Array, column, row)
    (height, width) = size(board)
    if 0 <= column < width && 0 <= row < height
      board[row + 1, column + 1] = modify_function(board[row + 1, column + 1])
    end
  end
end
#
# Applications
set_former_fighter = set_board_column_row(cell_value_set_former_fighter)
add_N_border = set_board_column_row(cell_value_add_N_border)
add_S_border = set_board_column_row(cell_value_add_S_border)
add_W_border = set_board_column_row(cell_value_add_W_border)
add_E_border = set_board_column_row(cell_value_add_E_border)
set_white = set_board_column_row(cell_value_set_white)
set_black = set_board_column_row(cell_value_set_black)
set_empty = set_board_column_row(cell_value_set_empty)
unset_former_fighter = set_board_column_row(cell_value_unset_former_fighter)
remove_N_border = set_board_column_row(cell_value_remove_N_border)
remove_S_border = set_board_column_row(cell_value_remove_S_border)
remove_W_border = set_board_column_row(cell_value_remove_W_border)
remove_E_border = set_board_column_row(cell_value_remove_E_border)

###################################
# ACTIONS                         #
###################################
#
# Actions are encoded as 0-2047 integers with bits set according to:
# Bits 10-3: coordinates CR of NW corner of origin bonbon (cell index from top to bottom, left to right)
const NUM_ACTIONS_PER_CELL = 0x8
# Bit 2: move type, 0 for division | 1 for fusion
const ACTION_TYPE_MASK = 0x4 # Bit encoding action type
# Bits 1-0: move direction, 0 for N | 1 for E | 2 for S | 3 for W
const ACTION_DIRECTION_DIV = 0x4
const DIRECTION_NORTH = 0x0
const DIRECTION_EAST = 0x1
const DIRECTION_SOUTH = 0x2
const DIRECTION_WEST = 0x3
#
# Build action value from coordinates of NW corner of bonbon, move type and direction
action_value(column, row, move_type, move_direction) = NUM_ACTIONS_PER_CELL * (BOARD_SIDE * column + row) + ACTION_TYPE_MASK * move_type + move_direction
#
const ACTIONS = Vector{Int}(collect(0:NUM_ACTIONS_PER_CELL * NUM_CELLS - 1))
#
GI.actions(::GameSpec) = ACTIONS
#
###
# Return an array of all potential actions for a given originating bonbon with
# NW corner at provided column and row.
###
# function origin_NW_corner_cell_index_to_potential_actions(cell_index::Int)
#   (column, row) = cell_index_to_column_row(cell_index)
#   if column >= BOARD_SIDE || row >= BOARD_SIDE
#     return nothing
#   end
#   # Set bits 10-3
#   coords::Int = NUM_ACTIONS_PER_CELL * (BOARD_SIDE * Int(column) + Int(row))
#   # Potential actions are all divisions and all fusions
#   # => create an item for all possible values of bits 2-0
#   return [coords + i for i in 0:NUM_ACTIONS_PER_CELL - 1]
# end
#  
# const ACTIONS = [
#     (
#       [
#         origin_NW_corner_cell_index_to_potential_actions(cell_index)
#         for cell_index in 1:NUM_CELLS
#       ]...
#     )...
# ]
#
# Finally potential actions are all integers up to NUM_ACTIONS_PER_CELL * NUM_CELLS
# for each cell and the encoded value of an action is 8 * (originating cell index) + (action code)

###################################
# BOARDS                          #
###################################
#
###
# Create the initial board object based on a list of  bonbons
# described in an array of bonbon notations
###
function bonbons_list_to_board(bonbons::Array)
  board = Array(ZERO_BOARD)
  try
    # Iterate on the array of bonbon notations to read borders and colors
    for bonbon in bonbons
      if length(bonbon) != 5
        return nothing
      else
        # Parse bonbon notation to get its properties
        west = parse(Int, bonbon[2], base=16)
        north = parse(Int, bonbon[3], base=16)
        east = parse(Int, bonbon[4], base=16)
        south = parse(Int, bonbon[5], base=16)
        team = bonbon[1] == '0' ? WHITE : BLACK
        # Add corresponding bits to bonbon interior and limit cells
        # NB. Columns and rows are expressed in a 0-based conceptual matrix
        #     whereas board is a 1-based Julia matrix
        for column in west:east
          for row in north:south
            # Add vertical border bits to vertical border cells and their horizontal neighbors
            if column == west
              add_W_border(board, column, row)
              add_E_border(board, column - 1, row)
            end
            if column == east
              add_E_border(board, column, row)
              add_W_border(board, column + 1, row)
            end
            # Add horizontal border bits to horizontal border cells and their vertical neighbors
            if row == north
              add_N_border(board, column, row)
              add_S_border(board, column, row - 1)
            end
            if row == south
              add_S_border(board, column, row)
              add_N_border(board, column, row + 1)
            end
            # Add team bit to bonbon cells
            if team == WHITE
              set_white(board, column, row)
            else
              set_black(board, column, row)
            end
          end
        end
      end
    end
    # Return the board
    return Board(board)
  catch
    return nothing
  end
end
#
const INITIAL_BOARD = bonbons_list_to_board(INITIAL_BOARD_SIZE_16_LIST)


###################################
# GAME RULES                      #
###################################
#
#=
Compute cell contribution to state properties which depend on the board data only.
Modify the actions hook and impact tables during the process.
Returns the legal actions attached to this cell, i.e. a 8-long boolean array to flag possible actions
from the point of view of the originating bonbon, if this cell is its NW corner.
The position of True values in the returned array and the subsequent value of a binary encoding
of this boolean series are illustrated below.
-----------------------
*W *S *E *N /W /S /E /N
 8  4  2  1  8  4  2  1 (legend for encoding the value of 1 action)
 8  7  6  5  4  3  2  1 (legend for the binary vector encoding all possible actions for a bonbon)
-----------------------
          1           1
       2           2   
       3  3        3  3
    4           4      
    5     5     5     5
    6  6        6  6   
    7  7  7     7  7  7
 8           8         
 9        9  9        9
 A     A     A     A   
 B     B  B  B     B  B
 C  C        C  C      
 D  D     D  D  D     D
 E  E  E     E  E  E   
 F  F  F  F  F  F  F  F
-----------------------
*W *S *E *N /W /S /E /N
 8  7  6  5  4  3  2  1 (legend for the binary vector encoding all possible actions for a bonbon)
 8  4  2  1  8  4  2  1 (legend for encoding the value of 1 action)
-----------------------
=#
function compute_cell_legal_actions_update_actions_hook_impact!(board::Board, column, row, actions_hook::Array, impact:: Array)
  if !(0 <= column < BOARD_SIDE) || !(0 <= row < BOARD_SIDE)
    return nothing
  end
  if is_empty(board, column, row)
    # Cell is empty => No legal actions, actions_hook undefined, empty impact.
    actions_hook[row + 1, column + 1] = (-1, -1)
    impact[row + 1, column + 1] = (0, 0)
    return falses(NUM_ACTIONS_PER_CELL)
  elseif !has_N_border(board, column, row) || !has_W_border(board, column, row)
    # Cell has no North border or no West border => No legal actions, empty impact, actions_hook was defined when
    # the NW corner of this bonbon was visited.
    impact[row + 1, column + 1] = (0, 0)
    return falses(NUM_ACTIONS_PER_CELL)
  end
  # (column, row) cell is a NW corner.
  # Initialize permissions
  can_fuse_N = can_fuse_E = can_fuse_S = can_fuse_W = true
  can_divide_N = can_divide_E = can_divide_S = can_divide_W = false
  # Initialize impact setup
  frontline = 0
  is_opponent = is_white(board, column, row) ? is_black : is_white
  # Is bonbon a former figher?
  bonbon_is_former_fighter = is_former_fighter(board, column, row)
  # Check if bonbon has a North neighbor and both aren't former fighters
  if is_empty(board, column, row - 1) || bonbon_is_former_fighter && is_former_fighter(board, column, row - 1)
    can_fuse_N = false
  end
  # Check the North border of the bonbon until the NE corner is reached
  last_column = column
  frontline += is_opponent(board, last_column, row - 1) ? 1 : 0
  while !has_E_border(board, last_column, row)
    last_column += 1
    frontline += is_opponent(board, last_column, row - 1) ? 1 : 0
    # If a cell has an vertical border then fusing to the North isn't legal
    if has_W_border(board, last_column, row - 1)
      can_fuse_N = false
    end
  end
  # last_column, row are now coordinates of the NE corner
  # Check if bonbon has a East neighbor and both aren't former fighters
  if is_empty(board, last_column + 1, row) || bonbon_is_former_fighter && is_former_fighter(board, last_column + 1, row)
    can_fuse_E = false
  end
  # Check the East border of the bonbon until the SE corner is reached
  last_row = row
  frontline += is_opponent(board, last_column + 1, last_row) ? 1 : 0
  while !has_S_border(board, last_column, last_row)
    last_row += 1
    frontline += is_opponent(board, last_column + 1, last_row) ? 1 : 0
    # If a cell has an horizontal border then fusing to the East isn't legal
    if has_N_border(board, last_column + 1, last_row)
      can_fuse_E = false
    end
  end
  # last_column, last_row are now coordinates of the SE corner
  # Check if bonbon has a South neighbor and both aren't former fighters
  if is_empty(board, last_column, last_row + 1) || bonbon_is_former_fighter && is_former_fighter(board, last_column, last_row + 1)
    can_fuse_S = false
  end
  # Check the South border of the bonbon
  frontline += is_opponent(board, column, last_row + 1) ? 1 : 0
  for i in column + 1:last_column
    frontline += is_opponent(board, i, last_row + 1) ? 1 : 0
    # If a cell has an vertical border then fusing to the South isn't legal
    if has_W_border(board, i, last_row + 1)
      can_fuse_S = false
    end
  end
  # Check if bonbon has a West neighbor and both aren't former fighters
  if is_empty(board, column - 1, row) || bonbon_is_former_fighter && is_former_fighter(board, column - 1, row)
    can_fuse_W = false
  end
  # Check the West border of the bonbon
  frontline += is_opponent(board, column - 1, row) ? 1 : 0
  for i in row + 1:last_row
    frontline += is_opponent(board, column - 1, i) ? 1 : 0
    # If a cell has an horizontal border then fusing to the West isn't legal
    if has_N_border(board, column - 1, i)
      can_fuse_W = false
    end
  end
  # Check legal division moves
  # Vertical divisions
  if last_row > row
    can_divide_S = true
    # There are 2 possible vertical directions only if there's an odd number of rows
    # ie difference is even
    can_divide_N = (row - last_row) % 2 == 0
  end
  # Horizontal divisions
  if last_column > column
    can_divide_E = true
    # There are 2 possible horizontal directions only if there's an odd number of columns
    # ie difference is even
    can_divide_W = (column - last_column) % 2 == 0
  end
  # Update actions_hook for all cells in the bonbon
  for c in column:last_column
    for r in row:last_row
      actions_hook[r + 1, c + 1] = (column, row)
    end
  end
  # Store the impact
  centrality = round((BOARD_SIDE - (abs(column + last_column - BOARD_SIDE + 1) / 2 + abs(row + last_row - BOARD_SIDE + 1) / 2)) * 6)
  impact[row + 1, column + 1] = (frontline , centrality)
  # Return cell's legal actions mask
  return [can_divide_N, can_divide_E, can_divide_S, can_divide_W, can_fuse_N, can_fuse_E, can_fuse_S, can_fuse_W]
end
#
# Compute state properties which depend on the board data only
function compute_actions_masks_actions_hooks_impact(b::Board)
  amask_white = []
  amask_black = []
  actions_hook = Array(EMPTY_TUPLE)
  impact = Array(ZERO_TUPLE)
  try
    for column in 0:BOARD_SIDE - 1
      for row in 0:BOARD_SIDE - 1
        cell_amask = compute_cell_legal_actions_update_actions_hook_impact!(b, column, row, actions_hook, impact)
        if is_white(b, column, row)
          append!(amask_white, cell_amask)
          append!(amask_black, falses(NUM_ACTIONS_PER_CELL))
        else
          append!(amask_black, cell_amask)
          append!(amask_white, falses(NUM_ACTIONS_PER_CELL))
        end
      end
    end
    return (amask_white, amask_black, BoardTuple(actions_hook), BoardTuple(impact))
  catch e
    println("Error in compute_actions_masks_actions_hooks_impact(b::Board)")
    println("Exception ", e)
    return nothing
  end
end
#
# Update the game status
function update_status!(g::GameEnv)
  (g.amask_white, g.amask_black, g.actions_hook, g.impact) = compute_actions_masks_actions_hooks_impact(g.board)
  g.amask = (g.curplayer == WHITE ? g.amask_white : g.amask_black)
  if any(g.amask)
    white_ratio = GI.heuristic_value(g)
    if white_ratio >= TO_CONQUER
      g.winner = WHITE
      g.finished = true
    elseif white_ratio <= 1 - TO_CONQUER
      g.winner = BLACK
      g.finished = true
    end
  else
    g.winner = 0
    g.finished = true
  end
end
#

function GI.play!(g::GameEnv, action)
  board = Array(g.board)
  # Define functions to move a cursor around the board
  step_N = (column, row) -> (column, row - 1)
  step_S = (column, row) -> (column, row + 1)
  step_W = (column, row) -> (column - 1, row)
  step_E = (column, row) -> (column + 1, row)
  # Unset former fighter bit for the whole board
  board = map(cell_value_unset_former_fighter, board)
  # Perform action on g
  if 0 <= action < NUM_ACTIONS_PER_CELL * NUM_CELLS
    try
      isnothing(g.history) || push!(g.history, action)
      (column, row) = (NW_column, NW_row) = index_to_column_row(action ÷ NUM_ACTIONS_PER_CELL + 1)
      action = action % NUM_ACTIONS_PER_CELL
      action_type = action & ACTION_TYPE_MASK
      action_direction = action % ACTION_DIRECTION_DIV
      if action_type == 0
        #
        # EXECUTE A DIVISION (add a border inside the bonbon)
        #
        # Select functions and variables to apply to the general division algorithm
        if action_direction in [DIRECTION_NORTH, DIRECTION_SOUTH]
          # Starting on bonbon's NW corner we always step South towards where to cut
          step_in_action_direction = step_S
          # Orthogonal direction relatively to the action direction
          step_in_orthog_direction = step_E
          # Test stopping borders
          has_closing_border_in_action_direction = has_S_border
          has_closing_border_in_orthog_direction = has_E_border
          # Add stopping borders
          add_closing_border_in_action_direction = add_S_border
          add_opening_border_in_action_direction = add_N_border
        else # action_direction in [DIRECTION_WEST, DIRECTION_EAST]
          # Starting on bonbon's NW corner we always step East towards where to cut
          step_in_action_direction = step_E
          # Orthogonal direction relatively to the action direction
          step_in_orthog_direction = step_S
          # Test stopping borders
          has_closing_border_in_action_direction = has_E_border
          has_closing_border_in_orthog_direction = has_S_border
          # Add stopping borders
          add_closing_border_in_action_direction = add_E_border
          add_opening_border_in_action_direction = add_W_border
        end
        # Apply the general division algorithm, starting from NW corner of origin bonbon
        (column, row) = (NW_column, NW_row)
        # Measure bonbon's dimension in the action direction
        dimension_in_action_direction = 1
        while !has_closing_border_in_action_direction(board, column, row)
          dimension_in_action_direction += 1
          (column, row) = step_in_action_direction(column, row)
        end
        # Return to the NW corner
        (column, row) = (NW_column, NW_row)
        # Move half of bonbon's dimension alongside the action direction, to position an
        # orthogonal border at the position rounded according to the the action direction
        # in case bonbon has an odd  dimension.
        for _ in 2:dimension_in_action_direction ÷ 2 + (action_direction in [DIRECTION_NORTH, DIRECTION_WEST] ? dimension_in_action_direction % 2 : 0)
          (column, row) = step_in_action_direction(column, row)
        end
        # Add a separation along the direction orthogonal to the action direction
        add_closing_border_in_action_direction(board, column, row)
        add_opening_border_in_action_direction(board, step_in_action_direction(column, row)...)
        while !has_closing_border_in_orthog_direction(board, column, row)
          (column, row) = step_in_orthog_direction(column, row)
          add_closing_border_in_action_direction(board, column, row)
          add_opening_border_in_action_direction(board, step_in_action_direction(column, row)...)
          # (The last addition won't have any effect if we step out of the board limits)
        end
      else
        #
        # EXECUTE A FUSION
        #
        # Select functions and variables to apply to the general fusion algorithm
        if action_direction == DIRECTION_NORTH
          step_in_action_direction = step_N
          step_in_action_reverse_direction = step_S
          # Does NW corner of origin bonbon touch destination bonbon?
          NW_touches_destination = true
          # Starting on bonbon's NW corner we measure bonbon's dimension in action direction
          # using South direction
          step_in_measured_action_direction = step_S
          # From where we arrived, measure bonbon's dimension in orthog direction
          # using East direction
          step_in_measured_orthog_direction = step_E
          # Orthogonal directions relatively to the action direction
          step_in_orthog_direction = step_E # Direction of 2nd cut
          step_in_orthog_reverse_direction = step_W # Reverse direction of 2nd cut
          # Test borders
          has_closing_border_in_measured_action_direction = has_S_border
          has_closing_border_in_measured_orthog_direction = has_E_border
          has_closing_border_in_action_direction = has_N_border
          has_closing_border_in_orthog_direction = has_E_border
          has_closing_border_in_orthog_reverse_direction = has_W_border
          # Add borders
          add_closing_border_in_action_direction = add_N_border
          add_opening_border_in_action_direction = add_S_border
          add_closing_border_in_orthog_direction = add_E_border
          add_opening_border_in_orthog_direction = add_W_border
          add_closing_border_in_orthog_reverse_direction = add_W_border
          add_opening_border_in_orthog_reverse_direction = add_E_border
          # Remove borders
          remove_closing_border_in_action_direction = remove_N_border
          remove_opening_border_in_action_direction = remove_S_border
        elseif action_direction == DIRECTION_SOUTH
          step_in_action_direction = step_S
          step_in_action_reverse_direction = step_N
          # Does NW corner of origin bonbon touch destination bonbon?
          NW_touches_destination = false
          # Starting on bonbon's NW corner we measure bonbon's dimension in action direction
          # using South direction
          step_in_measured_action_direction = step_S
          # From where we arrived, measure bonbon's dimension in orthog direction
          # using East direction
          step_in_measured_orthog_direction = step_E
          # Orthogonal directions relatively to the action direction
          step_in_orthog_direction = step_W # Direction of 2nd cut
          step_in_orthog_reverse_direction = step_E # Reverse direction of 2nd cut
          # Test borders
          has_closing_border_in_measured_action_direction = has_S_border
          has_closing_border_in_measured_orthog_direction = has_E_border
          has_closing_border_in_action_direction = has_S_border
          has_closing_border_in_orthog_direction = has_W_border
          has_closing_border_in_orthog_reverse_direction = has_E_border
          # Add borders
          add_closing_border_in_action_direction = add_S_border
          add_opening_border_in_action_direction = add_N_border
          add_closing_border_in_orthog_direction = add_W_border
          add_opening_border_in_orthog_direction = add_E_border
          add_closing_border_in_orthog_reverse_direction = add_E_border
          add_opening_border_in_orthog_reverse_direction = add_W_border
          # Remove borders
          remove_closing_border_in_action_direction = remove_S_border
          remove_opening_border_in_action_direction = remove_N_border
        elseif action_direction == DIRECTION_EAST
          step_in_action_direction = step_E
          step_in_action_reverse_direction = step_W
          # Does NW corner of origin bonbon touch destination bonbon?
          NW_touches_destination = false
          # Starting on bonbon's NW corner we measure bonbon's dimension in action direction
          # using East direction
          step_in_measured_action_direction = step_E
          # From where we arrived, measure bonbon's dimension in orthog direction
          # using North direction
          step_in_measured_orthog_direction = step_S
          # Orthogonal directions relatively to the action direction
          step_in_orthog_direction = step_N # Direction of 2nd cut
          step_in_orthog_reverse_direction = step_S # Reverse direction of 2nd cut
          # Test borders
          has_closing_border_in_measured_action_direction = has_E_border
          has_closing_border_in_measured_orthog_direction = has_S_border
          has_closing_border_in_action_direction = has_E_border
          has_closing_border_in_orthog_direction = has_N_border
          has_closing_border_in_orthog_reverse_direction = has_S_border
          # Add borders
          add_closing_border_in_action_direction = add_E_border
          add_opening_border_in_action_direction = add_W_border
          add_closing_border_in_orthog_direction = add_N_border
          add_opening_border_in_orthog_direction = add_S_border
          add_closing_border_in_orthog_reverse_direction = add_S_border
          add_opening_border_in_orthog_reverse_direction = add_N_border
          # Remove borders
          remove_closing_border_in_action_direction = remove_E_border
          remove_opening_border_in_action_direction = remove_W_border
        elseif action_direction == DIRECTION_WEST
          step_in_action_direction = step_W
          step_in_action_reverse_direction = step_E
          # Does NW corner of origin bonbon touch destination bonbon?
          NW_touches_destination = true
          # Starting on bonbon's NW corner we measure bonbon's dimension in action direction
          # using East direction
          step_in_measured_action_direction = step_E
          # From where we arrived, measure bonbon's dimension in orthog direction
          # using South direction
          step_in_measured_orthog_direction = step_S
          # Orthogonal directions relatively to the action direction
          step_in_orthog_direction = step_S # Direction of 2nd cut
          step_in_orthog_reverse_direction = step_N # Reverse direction of 2nd cut
          # Test borders
          has_closing_border_in_measured_action_direction = has_E_border
          has_closing_border_in_measured_orthog_direction = has_S_border
          has_closing_border_in_action_direction = has_W_border
          has_closing_border_in_orthog_direction = has_S_border
          has_closing_border_in_orthog_reverse_direction = has_N_border
          # Add borders
          add_closing_border_in_action_direction = add_W_border
          add_opening_border_in_action_direction = add_E_border
          add_closing_border_in_orthog_direction = add_S_border
          add_opening_border_in_orthog_direction = add_N_border
          add_closing_border_in_orthog_reverse_direction = add_N_border
          add_opening_border_in_orthog_reverse_direction = add_S_border
          # Remove borders
          remove_closing_border_in_action_direction = remove_W_border
          remove_opening_border_in_action_direction = remove_E_border
        end
        # Apply the general fusion algorithm, starting from NW corner of origin bonbon
        # ┼────────────────────────────┼          ┼──────────┼>>>>>>>>6>>>>>>>>┼
        # │                     Corner1│          │          ^          Corner2│
        # │      Dest                 ^│          │          ^                 │
        # │     (up or left)          ^3          │          6                 │
        # │                           ^│          │Final     ^                 │
        # │<<<<<<<4<<<<<<<<<┼----4-----┼────1─────┼-----8----┼>>>>>>>>7>>>>>>>>│
        # │ProjPiv1    Pivot¦P1<<<<<<<B│NW>>>>>>>>│<<<<<<<<P2¦Pivot    ProjPiv2│
        # │                 5v         │  Origin v2         5^                 │
        # │ProjPiv2    Pivot¦P2>>>>>>>>│        SE│B>>>>>>>P1¦Pivot    ProjPiv1│
        # │<<<<<<<7<<<<<<<<<┼----8-----┼──────────┼-----4----┼>>>>>>>>4>>>>>>>>│
        # │                 v     Final│          │v                           │
        # │                 6          │          │v          Dest             │
        # │                 v          │          3v          (down or right)  │
        # │Corner2          v          │          │Corner1                     │
        # ┼<<<<<<<6<<<<<<<<<┼──────────┼          ┼────────────────────────────┼
        # This diagram illustrates the following algorithm for both possible horizontal moves.
        # The representation also accounts for both possible vertical moves.
        #(1)
        # Measure origin bonbon's dimension in the action direction (which will be the cut depth)
        # using only East or South directions since the measure is done starting on from NW corner.
        dimension_in_action_direction = 1
        while !has_closing_border_in_measured_action_direction(board, column, row)
          dimension_in_action_direction += 1
          (column, row) = step_in_measured_action_direction(column, row)
        end
        cut_depth = dimension_in_action_direction
        # (2)
        # From where we are, measure origin bonbon's dimension in the orthogonal direction
        # (which will be the cut span).
        dimension_in_orthog_direction = 1
        while !has_closing_border_in_measured_orthog_direction(board, column, row)
          dimension_in_orthog_direction += 1
          (column, row) = step_in_measured_orthog_direction(column, row)
        end
        # Register SE corner of origin bonbon where we have arrived
        (SE_column, SE_row) = (column, row)
        # Define the base position, which is 1 cell away in action direction from either NW corner
        # or SE corner of origin bonbon.
        (column, row) = (base_column, base_row) = step_in_action_direction((NW_touches_destination ? (NW_column, NW_row) : (SE_column, SE_row))...)
        # Register a 1st corner of destination bonbon, which will be used to set the former fighter bit
        # and set actions hooks on its whole area.
        # (3)
        # From base position, look for a border in orthogonal reverse direction.
        while !has_closing_border_in_orthog_reverse_direction(board, column, row)
          (column, row) = step_in_orthog_reverse_direction(column, row)
        end
        (dest_corner1_column, dest_corner1_row) = (column, row)
        # 1st cut
        # Start penetration into destination bonbon from base position
        (column, row) = (base_column, base_row)
        # (4)
        # Traverse whole destination bonbon in action direction and add a separation between the visited
        # cells and the cells located 1 step away in the orthogonal reverse direction.
        # Also define a 1st pivot cell, where the penetrated area has same dimension than origin bonbon
        # in the action direction (or where destination bonbon border is reached in action direction)
        length_cut = 1
        add_closing_border_in_orthog_reverse_direction(board, column, row)
        add_opening_border_in_orthog_reverse_direction(board, step_in_orthog_reverse_direction(column, row)...)
        if length_cut == cut_depth || has_closing_border_in_action_direction(board, column, row)
          (pivot1_column, pivot1_row) = (column, row)
        end
        while !has_closing_border_in_action_direction(board, column, row)
          length_cut +=1
          (column, row) = step_in_action_direction(column, row)
          add_closing_border_in_orthog_reverse_direction(board, column, row)
          add_opening_border_in_orthog_reverse_direction(board, step_in_orthog_reverse_direction(column, row)...)
          # (The last addition won't have any effect if we step out of the board limits)
          if length_cut == cut_depth
            (pivot1_column, pivot1_row) = (column, row)
          end
        end
        # Adjust the cut depth and define 1st pivot position for the case where
        # origin bonbon is > destination bonbon in action dimension
        if length_cut < cut_depth
          cut_depth = length_cut
          (pivot1_column, pivot1_row) = (column, row)
        end
        # Store the projection of pivot1 on the furtherst border of destination bonbon for later use
        (pc1_column, pc1_row) = (column, row)
        # 2nd cut
        # (5)
        # Cut destination bonbon from 1st pivot position in orthogonal direction for the same dimension
        # as origin bonbon in this direction, adding a separation between the visited cells and the cells
        # located 1 step away in the action direction.
        (column, row) = (pivot1_column, pivot1_row)
        for _ in 1:dimension_in_orthog_direction
          # NB Since the action is deemed legal, the destination bonbon is "big enough" to be cut in this
          # dimension for this distance. No risk to reach a border before the end of the loop.
          add_closing_border_in_action_direction(board, column, row)
          add_opening_border_in_action_direction(board, step_in_action_direction(column, row)...)
          (column, row) = step_in_orthog_direction(column, row)
        end
        # Define a 2nd pivot cell where the penetrated area has same dimension than the origin bonbon
        # in the orthogonal direction
        (pivot2_column, pivot2_row) = step_in_orthog_reverse_direction(column, row)
        #
        # Store a 2nd corner of destination bonbon, which will be used to set the former fighter bit
        # on its whole area.
        # (6)
        # From 2nd pivot position, look for a border in orthogonal direction, then move in the
        # action direction for the distance we already know now all the way to the border.
        (column, row) = (pivot2_column, pivot2_row)
        while !has_closing_border_in_orthog_direction(board, column, row)
          (column, row) = step_in_orthog_direction(column, row)
        end
        for _ in cut_depth:length_cut - 1
          (column, row) = step_in_action_direction(column, row)
        end
        (dest_corner2_column, dest_corner2_row) = (column, row)
        # 3rd cut
        # From 2nd pivot position
        (column, row) = (pivot2_column, pivot2_row)
        # (7)
        # Traverse whole destination bonbon in action direction and add a separation between the visited
        # cells and the cells located 1 step away in the orthogonal direction.
        add_closing_border_in_orthog_direction(board, column, row)
        add_opening_border_in_orthog_direction(board, step_in_orthog_direction(column, row)...)
        for _ in cut_depth:length_cut - 1
          (column, row) = step_in_action_direction(column, row)
          add_closing_border_in_orthog_direction(board, column, row)
          add_opening_border_in_orthog_direction(board, step_in_orthog_direction(column, row)...)
          # (The last addition won't have any effect if we step out of the board limits)
        end
        # Store the projection of pivot2 on the furtherst border of destination bonbon for later use
        (pc2_column, pc2_row) = (column, row)
        # 4th cut
        # (8)
        # Back to 2nd pivot position, cut destination bonbon from there in action reverse direction
        # for the cut depth, and add a separation between the visited cells and the cells located
        # 1 step away in the orthogonal direction.
        (column, row) = (pivot2_column, pivot2_row)
        for _ in 1:cut_depth
          add_closing_border_in_orthog_direction(board, column, row)
          add_opening_border_in_orthog_direction(board, step_in_orthog_direction(column, row)...)
          (column, row) = step_in_action_reverse_direction(column, row)
        end
        # Store the final cell modified, for later use
        (final_column, final_row) = step_in_action_direction(column, row)
        #
        # Fuse
        # From base position, remove the separation between origin and destination bonbons
        (column, row) = (base_column, base_row)
        for _ in 1:dimension_in_orthog_direction
          remove_opening_border_in_action_direction(board, column, row)
          remove_closing_border_in_action_direction(board, step_in_action_reverse_direction(column, row)...)
          (column, row) = step_in_orthog_direction(column, row)
        end
        #
        # Register conquest
        # Attribute the conquered area from destination bonbon to origin bonbon's team.
        # Base and 2nd pivot positions form a diagonal of the area to paint.
        for column in min(base_column, pivot2_column):max(base_column, pivot2_column)
          for row in min(base_row, pivot2_row):max(base_row, pivot2_row)
            is_white(board, NW_column, NW_row) ? set_white(board, column, row) : set_black(board, column, row)
          end
        end
        #
        # Ensure a tuple represents coordinates of a cell which isn't out of the board limits
        in_boundaries = x -> max(0, min(BOARD_SIDE - 1, x))
        in_board = (column, row) -> (in_boundaries(column), in_boundaries(row))
        # Update former fighters bit over the extended origin bonbon and remaining pieces of destination bonbon
        function update_former_fighter((diag1_column, diag1_row), (diag2_column, diag2_row))
          for column in min(diag1_column, diag2_column):max(diag1_column, diag2_column)
            for row in min(diag1_row, diag2_row):max(diag1_row, diag2_row)
              set_former_fighter(board, column, row)
            end
          end
        end
        # For the extended origin bonbon, use the diagonal between pivot1 and:
        # - SE of origin bonbon, if NW of origin bonbon touches destination bonbon
        # - NW of origin bonbon, otherwise.
        (diag1_column, diag1_row) = (pivot1_column, pivot1_row)
        (diag2_column, diag2_row) = NW_touches_destination ? (SE_column, SE_row) : (NW_column, NW_row)
        update_former_fighter((diag1_column, diag1_row), (diag2_column, diag2_row))
        # For remaining pieces of destination bonbon area, up to 3 parts between previously defined cells
        # 1st part, if corner 1 is distinct from the base position, spans from corner1
        # to (1 step away from) the projection of pivot1 along 1st cut
        if (dest_corner1_column, dest_corner1_row) != (base_column, base_row)
          (diag1_column, diag1_row) = (dest_corner1_column, dest_corner1_row)
          (diag2_column, diag2_row) = in_board(step_in_orthog_reverse_direction(pc1_column, pc1_row)...)
          update_former_fighter((diag1_column, diag1_row), (diag2_column, diag2_row))
        end
        # 2nd part, if its projection is distinct from pivot1, spans from (1 step away from) pivot1
        # to the projection of pivot2 along 3rd cut
        if (pivot1_column, pivot1_row) != (pc1_column, pc1_row)
          (diag1_column, diag1_row) = in_board(step_in_action_direction(pivot1_column, pivot1_row)...)
          (diag2_column, diag2_row) = (pc2_column, pc2_row)
          update_former_fighter((diag1_column, diag1_row), (diag2_column, diag2_row))
        end
        # 3nd part, if corner2 is distinct from the projection of pivot2, spans from corner 2 to
        # the final cell modified during 4th cut
        if (dest_corner2_column, dest_corner2_row) != (pc2_column, pc2_row)
          (diag1_column, diag1_row) = (final_column, final_row)
          (diag2_column, diag2_row) = (dest_corner2_column, dest_corner2_row)
          update_former_fighter((diag1_column, diag1_row), (diag2_column, diag2_row))
        end
      end # Division or fusion is complete
      #
      # TIME FOR EXPANSION?
      #
      n_expansion = length(g.history) ÷ EXPAND_PERIOD
      if length(g.history) % EXPAND_PERIOD == 0 && 0 < n_expansion <= MAX_EXPANSIONS
        # Execute an expansion
        for column_h_borders in (n_expansion - 1):(BOARD_SIDE - n_expansion)
          # North border of the board
          row_h_borders = n_expansion - 1
          unset_former_fighter(board, column_h_borders, row_h_borders)
          set_empty(board, column_h_borders, row_h_borders)
          remove_N_border(board, column_h_borders, row_h_borders)
          remove_S_border(board, column_h_borders, row_h_borders - 1)
          remove_W_border(board, column_h_borders, row_h_borders)
          remove_E_border(board, column_h_borders - 1, row_h_borders)
          if !(column_h_borders in [n_expansion - 1, BOARD_SIDE - n_expansion])
            add_S_border(board, column_h_borders, row_h_borders)
            add_N_border(board, column_h_borders, row_h_borders + 1)
          end
          # West border of the board
          column_v_borders = row_h_borders
          row_v_borders = column_h_borders
          unset_former_fighter(board, column_v_borders, row_v_borders)
          set_empty(board, column_v_borders, row_v_borders)
          remove_W_border(board, column_v_borders, row_v_borders)
          remove_E_border(board, column_v_borders - 1, row_v_borders)
          remove_N_border(board, column_v_borders, row_v_borders)
          remove_S_border(board, column_v_borders, row_v_borders - 1)
          if !(row_v_borders in [n_expansion - 1, BOARD_SIDE - n_expansion])
            add_E_border(board, column_v_borders, row_v_borders)
            add_W_border(board, column_v_borders + 1, row_v_borders)
          end
          # East border of the board
          column_v_borders = BOARD_SIDE - row_h_borders - 1
          unset_former_fighter(board, column_v_borders, row_v_borders)
          set_empty(board, column_v_borders, row_v_borders)
          remove_E_border(board, column_v_borders, row_v_borders)
          remove_W_border(board, column_v_borders + 1, row_v_borders)
          remove_N_border(board, column_v_borders, row_v_borders)
          remove_S_border(board, column_v_borders, row_v_borders - 1)
          if !(row_v_borders in [n_expansion - 1, BOARD_SIDE - n_expansion])
            add_W_border(board, column_v_borders, row_v_borders)
            add_E_border(board, column_v_borders - 1, row_v_borders)
          end
          # South border of the board
          row_h_borders = BOARD_SIDE - n_expansion
          unset_former_fighter(board, column_h_borders, row_h_borders)
          set_empty(board, column_h_borders, row_h_borders)
          remove_S_border(board, column_h_borders, row_h_borders)
          remove_N_border(board, column_h_borders, row_h_borders + 1)
          remove_W_border(board, column_h_borders, row_h_borders)
          remove_E_border(board, column_h_borders - 1, row_h_borders)
          if !(column_h_borders in [n_expansion - 1, BOARD_SIDE - n_expansion])
            add_N_border(board, column_h_borders, row_h_borders)
            add_S_border(board, column_h_borders, row_h_borders - 1)
          end
        end # for column_h_borders
      end # Expansion is complete
    catch e
      println("Exception in GI.play! while executing action ", action, " on the following board:")
      GI.render(g)
    end
    #
    # Finalize game update
    #
    g.board = Board(board)
    g.curplayer = other_player(g.curplayer)
    update_status!(g)
  end
end


###################################
# HEURISTICS                      #
###################################
#
# Return the fraction of the non-empty board owned by the White player
# Used by the Minmax baseline player
function GI.heuristic_value(g::GameEnv)
  cells_count = NUM_CELLS - count(cell_value_is_empty, g.board)
  white_count = count(cell_value_is_white, g.board)
  return Float64(white_count / cells_count)
end
#
# Max number of actions to let available when applying heuristic to filter bonbons according to their potential impact
const HEURISTIC_MAX_ACTIONS = 10


###################################
# GAMES MANAGEMENT                #
###################################
#
(initial_amask_white, initial_amask_black, initial_actions_hook, initial_impact) = compute_actions_masks_actions_hooks_impact(INITIAL_BOARD)

# Create a game environment with initial conditions, defined in the list of bonbons INITIAL_BOARD_SIZE_16_LIST
function GI.init(::GameSpec)
  board = INITIAL_BOARD
  curplayer = WHITE
  actions_hook = copy(initial_actions_hook)
  amask_white = copy(initial_amask_white)
  amask_black = copy(initial_amask_black)
  amask = amask_white
  finished = false
  winner = 0x00
  history = Int[]
  impact = copy(initial_impact)
  g = GameEnv(board, curplayer, finished, winner, amask, amask_white, amask_black, actions_hook, history, impact)
  return g
end
#
# Constructor version which sets the initial state with hard-coded values which were pre-calculated starting with
# the list of bonbons INITIAL_BOARD_SIZE_16_LIST
# function Game()
#   s = read_state_array(["25.00.04.36 05.00.00.00 05.00.00.00 0d.00.00.00 25.40.08.48 05.40.00.00 05.40.00.00 0d.40.00.00 26.80.08.36 06.80.00.00 06.80.00.00 06.80.00.00 06.80.00.00 06.80.00.00 06.80.00.00 0e.80.00.00",
#     "21.00.00.00 01.00.00.00 01.00.00.00 09.00.00.00 21.40.00.00 01.40.00.00 01.40.00.00 09.40.00.00 22.80.00.00 02.80.00.00 02.80.00.00 02.80.00.00 02.80.00.00 02.80.00.00 02.80.00.00 0a.80.00.00",
#     "21.00.00.00 01.00.00.00 01.00.00.00 09.00.00.00 21.40.00.00 01.40.00.00 01.40.00.00 09.40.00.00 22.80.00.00 02.80.00.00 02.80.00.00 02.80.00.00 02.80.00.00 02.80.00.00 02.80.00.00 0a.80.00.00",
#     "21.00.00.00 01.00.00.00 01.00.00.00 09.00.00.00 31.40.00.00 11.40.00.00 11.40.00.00 19.40.00.00 32.80.00.00 12.80.00.00 12.80.00.00 12.80.00.00 12.80.00.00 12.80.00.00 12.80.00.00 1a.80.00.00",
#     "21.00.00.00 01.00.00.00 01.00.00.00 09.00.00.00 26.44.24.96 06.44.00.00 06.44.00.00 06.44.00.00 06.44.00.00 06.44.00.00 06.44.00.00 0e.44.00.00 25.c4.08.48 05.c4.00.00 05.c4.00.00 0d.c4.00.00",
#     "21.00.00.00 01.00.00.00 01.00.00.00 09.00.00.00 22.44.00.00 02.44.00.00 02.44.00.00 02.44.00.00 02.44.00.00 02.44.00.00 02.44.00.00 0a.44.00.00 21.c4.00.00 01.c4.00.00 01.c4.00.00 09.c4.00.00",
#     "21.00.00.00 01.00.00.00 01.00.00.00 09.00.00.00 22.44.00.00 02.44.00.00 02.44.00.00 02.44.00.00 02.44.00.00 02.44.00.00 02.44.00.00 0a.44.00.00 21.c4.00.00 01.c4.00.00 01.c4.00.00 09.c4.00.00",
#     "31.00.00.00 11.00.00.00 11.00.00.00 19.00.00.00 22.44.00.00 02.44.00.00 02.44.00.00 02.44.00.00 02.44.00.00 02.44.00.00 02.44.00.00 0a.44.00.00 31.c4.00.00 11.c4.00.00 11.c4.00.00 19.c4.00.00",
#     "25.08.08.48 05.08.00.00 05.08.00.00 0d.08.00.00 22.44.00.00 02.44.00.00 02.44.00.00 02.44.00.00 02.44.00.00 02.44.00.00 02.44.00.00 0a.44.00.00 25.c8.04.36 05.c8.00.00 05.c8.00.00 0d.c8.00.00",
#     "21.08.00.00 01.08.00.00 01.08.00.00 09.08.00.00 22.44.00.00 02.44.00.00 02.44.00.00 02.44.00.00 02.44.00.00 02.44.00.00 02.44.00.00 0a.44.00.00 21.c8.00.00 01.c8.00.00 01.c8.00.00 09.c8.00.00",
#     "21.08.00.00 01.08.00.00 01.08.00.00 09.08.00.00 22.44.00.00 02.44.00.00 02.44.00.00 02.44.00.00 02.44.00.00 02.44.00.00 02.44.00.00 0a.44.00.00 21.c8.00.00 01.c8.00.00 01.c8.00.00 09.c8.00.00",
#     "31.08.00.00 11.08.00.00 11.08.00.00 19.08.00.00 32.44.00.00 12.44.00.00 12.44.00.00 12.44.00.00 12.44.00.00 12.44.00.00 12.44.00.00 1a.44.00.00 21.c8.00.00 01.c8.00.00 01.c8.00.00 09.c8.00.00",
#     "26.0c.08.36 06.0c.00.00 06.0c.00.00 06.0c.00.00 06.0c.00.00 06.0c.00.00 06.0c.00.00 0e.0c.00.00 25.8c.08.48 05.8c.00.00 05.8c.00.00 0d.8c.00.00 21.c8.00.00 01.c8.00.00 01.c8.00.00 09.c8.00.00",
#     "22.0c.00.00 02.0c.00.00 02.0c.00.00 02.0c.00.00 02.0c.00.00 02.0c.00.00 02.0c.00.00 0a.0c.00.00 21.8c.00.00 01.8c.00.00 01.8c.00.00 09.8c.00.00 21.c8.00.00 01.c8.00.00 01.c8.00.00 09.c8.00.00",
#     "22.0c.00.00 02.0c.00.00 02.0c.00.00 02.0c.00.00 02.0c.00.00 02.0c.00.00 02.0c.00.00 0a.0c.00.00 21.8c.00.00 01.8c.00.00 01.8c.00.00 09.8c.00.00 21.c8.00.00 01.c8.00.00 01.c8.00.00 09.c8.00.00",
#     "32.0c.00.00 12.0c.00.00 12.0c.00.00 12.0c.00.00 12.0c.00.00 12.0c.00.00 12.0c.00.00 1a.0c.00.00 31.8c.00.00 11.8c.00.00 11.8c.00.00 19.8c.00.00 31.c8.00.00 11.c8.00.00 11.c8.00.00 19.c8.00.00",
#     "Pink"])
#   board = s.board
#   curplayer = s.curplayer
#   actions_hook = s.actions_hook
#   amask_white = string_to_booleans("62000000000000006e00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000670000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000006d000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000006b0000006800000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000")
#   amask_black = string_to_booleans("00000000000000000000000064000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000006000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000006100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000")
#   amask = amask_white
#   finished = false
#   winner = 0x00
#   history = Int[]
#   impact = s.impact
#   g = Game(board, curplayer, finished, winner, amask, amask_white, amask_black, actions_hook, history, impact)
#   return g
# end
#
# Set a given state. Consistency is ensured by updating the status.
function GI.set_state!(g::GameEnv, s)
  g.board = s.board
  g.curplayer = s.curplayer
  g.actions_hook = s.actions_hook
  g.amask_white = []
  g.amask_black = []
  g.amask = []
  g.finished = false
  g.winner = 0x00
  g.history = []
  g.impact = s.impact
  update_status!(g)
  return g
end
#
# Customization of the standard GI.actions_mask function: we added a use_heuristic argument in order to speed up
# baseline players by reducing the return set of available actions using the Impact Heuristic.
# This heuristic allows only actions from the most impactful bonbons, impact being calculated using the frontline
# and centrality properties of a bonbon:
# - frontline measures the number of opposing neighboring cells,
# - centrality is higher as the bonbon center is closer to the center of the board.
function GI.actions_mask(g::GameEnv, use_heuristic::Bool=false)
  amask = g.amask
  if use_heuristic
    bonbons = []
    for column in 0:BOARD_SIDE - 1
      for row in 0:BOARD_SIDE - 1
        tuple_impact = g.impact[row + 1, column + 1]
        if tuple_impact != (0, 0)
          # The bonbon at (column, row) should be considered
          # Determine bonbon impact: frontline * centrality / 2
          impact = round(tuple_impact[1] * tuple_impact[2] / 10)
          # Count bonbon actions
          first_action = (column_row_to_index((column, row)) - 1) * NUM_ACTIONS_PER_CELL + 1
          num_actions = count(==(true), amask[first_action:first_action + NUM_ACTIONS_PER_CELL - 1])
          # Record bonbon in the list
          push!(bonbons, ((column, row), impact, num_actions))
        end
      end
    end
    sort!(bonbons, by=(x -> x[2]), rev=true)
    last_index = num_actions = 0
    while num_actions < HEURISTIC_MAX_ACTIONS && last_index < length(bonbons)
      last_index += 1
      num_actions += bonbons[last_index][3]
    end
    for bonbon in bonbons[last_index + 1:end]
      first_action = (column_row_to_index(bonbon[1]) - 1) * NUM_ACTIONS_PER_CELL + 1
      for a in 0:NUM_ACTIONS_PER_CELL - 1
        amask[first_action + a] = false
      end
    end
  end
  return amask
end
#
# GI.actions_mask(g::GameEnv) = g.amask
GI.two_players(::GameSpec) = true
GI.current_state(g::GameEnv) = (board=g.board, impact=g.impact, actions_hook=g.actions_hook, curplayer=g.curplayer)
GI.white_playing(g::GameEnv) = g.curplayer == WHITE

#
function GI.clone(g::GameEnv)
  history = isnothing(g.history) ? nothing : copy(g.history)
  GameEnv(copy(g.board), g.curplayer, g.finished, g.winner, copy(g.amask), copy(g.amask_white), copy(g.amask_black), copy(g.actions_hook), history, copy(g.impact))
end
#
###
# Reward shaping
###
#
function GI.game_terminated(g::GameEnv)
  return g.finished
end
#
function GI.white_reward(g::GameEnv)
  if g.finished
    g.winner == WHITE && (return  1.)
    g.winner == BLACK && (return -1.)
    return 0.
  else
    return 0.
  end
end


###################################
# MACHINE LEARNING API            #
###################################
#
function flip_cell_color(c::Cell)
  if cell_value_is_empty(c)
    return c
  elseif cell_value_is_white(c)
    return cell_value_set_black(c)
  elseif cell_value_is_black(c)
    return cell_value_set_white(c)
  end
end
#
function flip_colors(board)
  return @SMatrix [
    flip_cell_color(board[row + 1, column + 1])
    for column in 0:BOARD_SIDE - 1, row in 0:BOARD_SIDE - 1]
end
#
# Vectorized representation: NUM_CELLS x NUM_CELLS x 6 array.
# Channels: empty, white, black, North border, West border, former fighter.
# The board is represented from the perspective of white (ie as if white were to play next)
function GI.vectorize_state(::GameSpec, state)
  board = state.curplayer == WHITE ? state.board : flip_colors(state.board)
  return Float32[
    property_cell(board, column, row)
    for column in 0:BOARD_SIDE - 1,
      row in 0:BOARD_SIDE - 1,
        property_cell in [is_empty, is_white, is_black,
          has_N_border, has_E_border, has_S_border, has_W_border, is_former_fighter
        ]
  ]
end

###################################
# SYMMETRIES                      #
###################################
#
# 90° rotation (anti clockwise)
rotate_cell((column, row)) = (row, BOARD_SIDE - 1 - column)
rotate_cell(value::Cell) = Cell(booleans_to_integer([cell_value_is_white(value), cell_value_is_black(value),
  cell_value_has_E_border(value), cell_value_has_S_border(value), cell_value_has_W_border(value), cell_value_has_N_border(value),
  cell_value_is_former_fighter(value)]))
function rotate_action(action::Int)
  # New coordinates of source cell
  (column, row) = rotate_cell(index_to_column_row(action ÷ NUM_ACTIONS_PER_CELL + 1))
  # Type (unchanged)
  type = (action & ACTION_TYPE_MASK) ÷ ACTION_TYPE_MASK
  # New direction of action
  direction = (action % ACTION_DIRECTION_DIV + 3) % ACTION_DIRECTION_DIV
  return action_value(column, row, type, direction)
end
# -90° rotation (clockwise)
inv_rotate_cell((column, row)) = (row, BOARD_SIDE - column)
inv_rotate_cell(value::Cell) = Cell(booleans_to_integer([cell_value_is_white(value), cell_value_is_black(value),
  cell_value_has_W_border(value), cell_value_has_N_border(value), cell_value_has_E_border(value), cell_value_has_S_border(value),
  cell_value_is_former_fighter(value)]))
function inv_rotate_action(action::Int)
  # New coordinates of source cell
  (column, row) = inv_rotate_cell(index_to_column_row(action ÷ NUM_ACTIONS_PER_CELL + 1))
  # Type (unchanged)
  type = (action & ACTION_TYPE_MASK) ÷ ACTION_TYPE_MASK
  # New direction of action
  direction = (action % ACTION_DIRECTION_DIV + 1) % ACTION_DIRECTION_DIV
  return action_value(column, row, type, direction)
end
#
# 180° rotations
rotate_cell_2 = rotate_cell ∘ rotate_cell
rotate_action_2 = rotate_action ∘ rotate_action
#
# 270° rotations
rotate_cell_3 = rotate_cell_2 ∘ rotate_cell
rotate_action_3 = rotate_action_2 ∘ rotate_action
#
# flip along horizontal axis
h_flip_cell((column, row)) = (column, BOARD_SIDE - row - 1)
h_flip_cell(value::Cell) = Cell(booleans_to_integer([cell_value_is_white(value), cell_value_is_black(value),
  cell_value_has_S_border(value), cell_value_has_E_border(value), cell_value_has_N_border(value), cell_value_has_W_border(value),
  cell_value_is_former_fighter(value)]))
function h_flip_action(action::Int)
  # New coordinates of former NW corner (which isn't NW corner any longer)
  (column, row) = h_flip_cell(index_to_column_row(action ÷ NUM_ACTIONS_PER_CELL + 1))
  # Type (unchanged)
  type = (action & ACTION_TYPE_MASK) ÷ ACTION_TYPE_MASK
  # New direction
  dir = (action % ACTION_DIRECTION_DIV)
  direction = dir % 2 == 0 ? (dir + 2) % ACTION_DIRECTION_DIV : dir
  return action_value(column, row, type, direction)
end
#
# Define non-identical symmetries as an array of tuples (cell_sym, inv_cell_sym, action_sym, inv_cell_sym)
const SYMMETRIES = [
  (rotate_cell, rotate_cell_3, rotate_action, rotate_action_3),
  (rotate_cell_2, rotate_cell_2, rotate_action_2, rotate_action_2),
  (rotate_cell_3, rotate_cell, rotate_action_3, rotate_action),
  (h_flip_cell, h_flip_cell, h_flip_action, h_flip_action),
  (h_flip_cell ∘ rotate_cell, rotate_cell_3 ∘ h_flip_cell, h_flip_action ∘ rotate_action, rotate_action_3 ∘ h_flip_action),
  (h_flip_cell ∘ rotate_cell_2, rotate_cell_2 ∘ h_flip_cell, h_flip_action ∘ rotate_action_2, rotate_action_2 ∘ h_flip_action),
  (h_flip_cell ∘ rotate_cell_3, rotate_cell ∘ h_flip_cell, h_flip_action ∘ rotate_action_3, rotate_action ∘ h_flip_action)
]
#=
Given a state state1 and a symmetry (sym_cell, inv_sym_cell, sym_action(not used here), inv_sym_action)
for cells and actions, return the pair (state2, σ) where state2 is the image of state1 by the symmetry
and σ is the associated actions permutations -an integer vector of size num_actions(GameSpec), such as
if actions_mask1 corresponds to state1 and actions_mask2 corresponds to state2:
actions_mask2[action_index] == actions_mask1[σ(action_index)]
=#
function apply_symmetry(state, sym_cell, inv_sym_cell, sym_action, inv_sym_action)
  board = state.board
  impact = state.impact
  actions_hook = state.actions_hook
  # Assess if applying the symmetry to a valid division along an evenly long bonbon dimension
  # could transform the action into an invalid division
  sym_of_unique_legal_div_could_be_N = inv_sym_action(action_value(0, 0, 0, DIRECTION_NORTH)) % ACTION_TYPE_MASK in [DIRECTION_EAST, DIRECTION_SOUTH]
  sym_of_unique_legal_div_could_be_W = inv_sym_action(action_value(0, 0, 0, DIRECTION_WEST)) % ACTION_TYPE_MASK in [DIRECTION_EAST, DIRECTION_SOUTH]
  # Apply the symmetry to the board, the impact table and the actions hook table
  res_board = Array(ZERO_BOARD)
  res_impact = Array(ZERO_TUPLE)
  res_actions_hook = Array(EMPTY_TUPLE)
  for column in 0:BOARD_SIDE - 1
    for row in 0:BOARD_SIDE - 1
      # sym_cell moves cell (column, row) to (newColumn, newRow) in the board, the impact and the actions hook tables...
      (newColumn, newRow) = sym_cell((column, row))
      res_impact[newRow + 1, newColumn + 1] = impact[row + 1, column + 1]
      # ...also changing the cells values for the board and the actions hook table
      res_board[newRow + 1, newColumn + 1] = sym_cell(board[row + 1, column + 1])
      res_actions_hook[newRow + 1, newColumn + 1] = sym_cell(actions_hook[row + 1, column + 1])
      # (The new actions hook position won't be on a NW corner any longer)
    end
  end # Symmetry is applied on a per cell basis
  #=
  Permutation of actions defined on a per cell basis:
  mask2[action_index] == mask1[σ(action_index)]
  mask2[action + 1] == mask1[σ(action + 1)]
  mask2[action + 1] == mask1[j + 1] with action == sym_action(j) i.e. j == inv_sym_action(action)
  mask2[action + 1] == mask1[inv_sym_action(action) + 1]
  Hence σ(action + 1) == inv_sym_action(action) + 1
  Hence σ(action_index) == inv_sym_action(action_index - 1) + 1
  =#
  σ = map(i -> inv_sym_action(i - 1) + 1, collect(1:NUM_ACTIONS_PER_CELL * NUM_CELLS))
  #=
  For the symmetry to work on a per bonbon basis while keeping the number of possible actions minimum:
  1. impact should be attached to post-symmetry NW corners
  2. σ should link the actions based on post-symmetry NW corners to the actions based on pre-symmetry
     NW corners of the same bonbons,
  3. actions hooks should point to post-symmetry NW corners for all cells in a bonbon,
  4. division actions pointing North or West along evenly long borders should be re-oriented.
  =#
  for column in 0:BOARD_SIDE - 1
    for row in 0:BOARD_SIDE - 1
      if has_N_border(res_board, column, row) && has_W_border(res_board, column, row)
        # Cell (column, row) is a post-sym NW corner.
        # 1. Get the impact from the transformed pre-sym NW corner
        # Locate the pre-sym NW corner
        (sym_of_pre_sym_NW_column, sym_of_pre_sym_NW_row) = res_actions_hook[row + 1, column + 1]
        # Swap fronline length values
        (res_impact[row + 1, column + 1], res_impact[sym_of_pre_sym_NW_row + 1, sym_of_pre_sym_NW_column + 1]) = (res_impact[sym_of_pre_sym_NW_row + 1, sym_of_pre_sym_NW_column + 1], res_impact[row + 1, column + 1])
        # 2. Adjust σ to point to the cell holding the pre-sym actions
        # The legal actions are the transformed actions of the pre-sym NW corner of its bonbon.
        # Locate the pre-sym NW corner of its bonbon: it transforms into the cell currently
        # pointed to in the transformed actions hook table.
        (pre_sym_NW_column, pre_sym_NW_row) = inv_sym_cell((sym_of_pre_sym_NW_column, sym_of_pre_sym_NW_row))
        # Make sure the permutation swaps the post-sym NW corner actions and the pre-sym NW corner ones,
        # the latter being (pre-sym) the only legal actions for the whole bonbon.
        # Adapt the permutation for the post-sym NW corner so that it points to pre-sym NW corner actions
        base_post_sym_index = NUM_ACTIONS_PER_CELL * (column_row_to_index((column, row)) - 1)
        delta_permutation = NUM_ACTIONS_PER_CELL * (column_row_to_index((sym_of_pre_sym_NW_column, sym_of_pre_sym_NW_row)) - 1) - base_post_sym_index
        for action_index in (base_post_sym_index + 1):(base_post_sym_index + NUM_ACTIONS_PER_CELL)
          # First, translate to the transformed pre-sym NW corner actions since this doesn't affect their order.
          # Then apply inv_sym_action to link with the actions of the pre-sym NW corner.
          σ[action_index] = inv_sym_action(action_index + delta_permutation - 1) + 1
        end
        # Adapt the permutation for the transformed pre-sym NW corner so that it points to
        # post-sym NW corner's pre-sym actions
        base_post_sym_index = NUM_ACTIONS_PER_CELL * (column_row_to_index((sym_of_pre_sym_NW_column, sym_of_pre_sym_NW_row)) - 1)
        delta_permutation = -delta_permutation
        for action_index in (base_post_sym_index + 1):(base_post_sym_index + NUM_ACTIONS_PER_CELL)
          # First, translate to the post-sym NW corner actions since this doesn't affect their order.
          # Then apply inv_sym_action to link with the actions of the naturally permuted post-sym NW corner.
          σ[action_index] = inv_sym_action(action_index + delta_permutation - 1) + 1
        end
        # 3. Adjust actions_hook
        # Update the actions hook for the whole bonbon in order to point to the new NW corner (column, row).
        # Determine the dimensions of this bonbon for which the new actions hook must be set
        width = height = 1
        while column + width < BOARD_SIDE && res_actions_hook[row + 1, column + 1 + width] == (sym_of_pre_sym_NW_column, sym_of_pre_sym_NW_row)
          width += 1
        end
        while row + height < BOARD_SIDE && res_actions_hook[row + 1 + height, column + 1] == (sym_of_pre_sym_NW_column, sym_of_pre_sym_NW_row)
          height += 1
        end
        # Update the actions hook for the whole bonbon area to point to the new NW corner
        for c in column:column + width - 1
          for r in row:row + height - 1
            res_actions_hook[r + 1, c + 1] = (column, row)
          end
        end # Actions hook table is updated
        # 4. Adjust σ to re-orient division actions
        if height % 2 == 0 && sym_of_unique_legal_div_could_be_N
          #= Height has even length => avoid divisions towards N created by natural application of  symmetry
          by swapping them in σ with divisions towards S, based on the assumption that in such configuration,
          if an action towards E or S transforms into an action towards N, then the actions in the other
          potentially unique direction (S or E, respectively) won't tranform in actions towards either N or S.
          =#
          σ[action_value(column, row, 0, DIRECTION_NORTH) + 1], σ[action_value(column, row, 0, DIRECTION_SOUTH) + 1] = σ[action_value(column, row, 0, DIRECTION_SOUTH) + 1], σ[action_value(column, row, 0, DIRECTION_NORTH) + 1]
        end
        if width % 2 == 0 && sym_of_unique_legal_div_could_be_W
          #= Width has even length => avoid divisions towards W created by natural application of  symmetry
          by swapping them in σ with divisions towards E, based on the assumption that in such configuration,
          if an action towards E or S transforms into an action towards W, then the actions in the other
          potentially unique direction (S or E, respectively) won't tranform in actions towards either W or E.
          =#
          σ[action_value(column, row, 0, DIRECTION_WEST) + 1], σ[action_value(column, row, 0, DIRECTION_EAST) + 1] = σ[action_value(column, row, 0, DIRECTION_EAST) + 1], σ[action_value(column, row, 0, DIRECTION_WEST) + 1]
        end
      end # New NW corner is processed
    end
  end # New board has been scanned
  # Return (s, σ)
  return (
    (board=Board(res_board), impact=BoardTuple(res_impact), actions_hook=BoardTuple(res_actions_hook), curplayer=state.curplayer),
    Vector{Int}(σ)
  )
end
#
function GI.symmetries(::GameSpec, s)
  return [apply_symmetry(s, sym_cell, inv_sym_cell, sym_action, inv_sym_action) for (sym_cell, inv_sym_cell, sym_action, inv_sym_action) in SYMMETRIES]
end

###################################
# USER INTERFACE                  #
###################################
#
GI.action_string(::GameSpec, a) = string(string(a ÷ NUM_ACTIONS_PER_CELL, base=16), " $(a & ACTION_TYPE_MASK == ACTION_TYPE_MASK ? '*' : '/') $(['N', 'E', 'S', 'W'][a % ACTION_DIRECTION_DIV + 1])")
#
function GI.parse_action(::GameSpec, str)
  try
    if length(str) == 4
      action_array = [str[1:2], str[3:3], str[4:4]]
    else
      action_array = split(str, ' ')
    end
    if length(action_array) != 3 return "length(action_array) != 3" end
    action_origin = parse(Int, action_array[1], base=16)
    if !(0 <= action_origin < NUM_CELLS) return "!(0 <= action_origin < NUM_CELLS)" end
    action_type = findfirst(isequal(action_array[2]), ["/", "*"]) - 1
    action_direction = findfirst(contains(action_array[3]), ["nN", "eE", "sS", "wW"]) - 1
    return action_origin * NUM_ACTIONS_PER_CELL + action_type * ACTION_TYPE_MASK + action_direction
  catch
    println("Error while parsing action '", str, "'")
    nothing
  end
end
#
# FOR DEBUG PURPOSE ONLY
function play_move!(g::GameEnv, s::String)
  GI.play!(g, GI.parse_action(GameSpec, s))
end
#
# The board is represented by a "print matrix" in which each cell represents either a piece of a board
# cell or a piece of border. Their encoding model is the same as for board cells (see CELLS section).
# Hence, the functions to test and change board cells values apply to the print matrix cells values.
# There's 1 row of border cells for 2 rows of "pulp" cells, 1 column of border cells for 2 columns of pulp cells.
#=        0  1  2  3  4  5  6  7  8  9  A  B  C  D  E  F 
        BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB
      0 BPPBPPBPPBPPBPPBPPBPPBPPBPPBPPBPPBPPBPPBPPBPPBPPB
        BPPBPPBPPBPPBPPBPPBPPBPPBPPBPPBPPBPPBPPBPPBPPBPPB
        BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB
      1 BPPBPPBPPBPPBPPBPPBPPBPPBPPBPPBPPBPPBPPBPPBPPBPPB
        BPPBPPBPPBPPBPPBPPBPPBPPBPPBPPBPPBPPBPPBPPBPPBPPB
        BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB
        .................................................
        .................................................
        .................................................
      F BPPBPPBPPBPPBPPBPPBPPBPPBPPBPPBPPBPPBPPBPPBPPBPPB
        BPPBPPBPPBPPBPPBPPBPPBPPBPPBPPBPPBPPBPPBPPBPPBPPB
        BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB
=#
const EMPTY_PRINT_MATRIX = @SMatrix zeros(Cell, 3BOARD_SIDE + 1, 3BOARD_SIDE + 1)
const PrintMatrix = typeof(EMPTY_PRINT_MATRIX)
#
column_row_to_print_matrix_pulp_corners(column, row) = (north_west=(row=3row+1, column=3column+1), south_east=(row=3row+2, column=3column+2))
column_row_to_print_matrix_N_border_corners(column, row) = (north_west=(row=3row, column=3column), south_east=(row=3row, column=3column+3))
column_row_to_print_matrix_S_border_corners(column, row) = (north_west=(row=3row+3, column=3column), south_east=(row=3row+3, column=3column+3))
column_row_to_print_matrix_W_border_corners(column, row) = (north_west=(row=3row, column=3column), south_east=(row=3row+3, column=3column))
column_row_to_print_matrix_E_border_corners(column, row) = (north_west=(row=3row, column=3column+3), south_east=(row=3row+3, column=3column+3))
#
# Gives a property of the board cell in (column, row) to the print matrix cells in an area.
# The property modifier is the argument function set_property.
# The corners of the area are provided by the argument function area_corners applied on column, row.
function print_cell_property_to_print_matrix!(column, row, print_matrix, area_corners, set_property)
  (northwest, southeast) = area_corners(column, row)
  for r in northwest.row:southeast.row
    for c in northwest.column:southeast.column
      set_property(print_matrix, c, r)
    end
  end
end
#
function build_print_matrix(g::GameEnv)
  board = g.board
  print_matrix = Array(EMPTY_PRINT_MATRIX)
  for column = 0:BOARD_SIDE - 1
    for row = 0:BOARD_SIDE - 1
      cell_value = board[row + 1, column + 1]
      for (test_cell_value, area_corners, set_property) in [
          (cell_value_has_N_border, column_row_to_print_matrix_N_border_corners, add_N_border),
          (cell_value_has_E_border, column_row_to_print_matrix_E_border_corners, add_E_border),
          (cell_value_has_S_border, column_row_to_print_matrix_S_border_corners, add_S_border),
          (cell_value_has_W_border, column_row_to_print_matrix_W_border_corners, add_W_border),
          (cell_value_is_white, column_row_to_print_matrix_pulp_corners, set_white),
          (cell_value_is_black, column_row_to_print_matrix_pulp_corners, set_black),
          (cell_value_is_former_fighter, column_row_to_print_matrix_pulp_corners, set_former_fighter)
      ]
        test_cell_value(cell_value) && print_cell_property_to_print_matrix!(column, row, print_matrix, area_corners, set_property)
      end
    end
  end
  return PrintMatrix(print_matrix)
end
#
player_color(p) = p == WHITE ? crayon"fg:white bg:magenta" : crayon"fg:white bg:green"
player_name(p)  = p == WHITE ? "Pink" : "Green"
#
function cell_color(c)
  if cell_value_has_N_border(c) || cell_value_has_E_border(c) || cell_value_has_S_border(c) || cell_value_has_W_border(c)
    return crayon"fg:white bg:black"
  elseif cell_value_is_white(c)
    return cell_value_is_former_fighter(c) ? crayon"fg:black bg:light_magenta" : crayon"fg:white bg:magenta"
  elseif cell_value_is_black(c)
    return cell_value_is_former_fighter(c) ? crayon"fg:black bg:light_green" : crayon"fg:white bg:green"
  else
    return crayon""
  end
end
#
# FOR DEBUG PURPOSE ONLY
# Display a print matrix
function print_print_matrix(print_matrix::PrintMatrix)
  # Print column labels
  print("  ")
  for c in 1:3BOARD_SIDE + 1
    print(" ", string(c, pad=2))
  end
  print("\n")
  # Print board
  for r in 1:3BOARD_SIDE + 1
    # Print row label
    print(string(r, pad=2), " ")
    for c in 1:3BOARD_SIDE + 1
      print_value = print_matrix[r, c]
      print(string(print_value, base=16, pad=2), " ")
    end # Row is completely printed
    print("\n")
  end
end
#
# Print the board to standard output
# The botmargin argument is used to display bonbons impact instead of coordinates and legal actions
function GI.render(g::GameEnv; with_position_names=true, botmargin=false)
  pname = player_name(g.curplayer)
  pcol = player_color(g.curplayer)
  print("Move ", length(g.history) + 1, ". ", pcol, pname, " plays:", crayon"reset", "\n\n")
  board = g.board
  amask = g.amask
  impact = g.impact
  print_matrix = build_print_matrix(g)
  print(" ")
  # Print column labels
  for c in 0:BOARD_SIDE - 1
    print("    ", string(c, base=16))
  end
  print("\n")
  # Print board
  for r in 1:3BOARD_SIDE + 1
    row = (r - 1) ÷ 3
    if (r - 2) % 3 == 0
      # Print row label
      print(string(row, base=16))
    else
      print(" ")
    end
    print(" ")
    for c in 1:3BOARD_SIDE + 1
      column = (c - 1) ÷ 3
      if botmargin && column < BOARD_SIDE && row < BOARD_SIDE
        frontline = impact[row + 1, column + 1][1]
        centrality = impact[row + 1, column + 1][2]
      end
      print_value = print_matrix[r, c]
      # Determine which char to print if not " " (background) :
      # If border cell:  graphic char with horizontal and/or vertical dash
      # If pulp cell: NW coordinates and encoding of legal actions on NW corner
      if cell_value_has_N_border(print_value) || cell_value_has_E_border(print_value) || cell_value_has_S_border(print_value) || cell_value_has_W_border(print_value)
        # Cell is a border cell, convert it to graphical char
        print_mark = ["─", "│", "┼", "─", "─", "┼", "┼", "│", "┼", "│", "┼", "┼", "┼", "┼", "┼"][(print_value % CELL_FORMER_FIGHTER) ÷ CELL_H_BORDER_NORTH]
        if (c - 1) % 3 in [1, 2]
          print_mark = print_mark * print_mark
        end
      elseif print_value == 0 # Cell is empty
        print_mark = " "
        if (c - 1) % 3 in [1, 2]
          print_mark = print_mark * print_mark
        end
      else # Cell is a pulp cell
        if has_N_border(board, column, row) && has_W_border(board, column, row)
          # Corresponding board cell is a NW corner => print coordinates and legal actions code,
          # unless botmargin=true then print bonbon impact
          if ((r - 1) % 3) == 1
            # North pulp cells of NW board cell => print coordinates or frontline
            if ((c - 1) % 3) == 1
              # NW pulp cell of NW board cell => print column or frontline 1st digit
              print_mark = " " * (botmargin ? string(frontline ÷ 10) : string(column, base=16))
            else
              # NE pulp cell of NW board cell => print row or frontline 2nd digit
              print_mark = (botmargin ? string(frontline % 10) : string(row, base=16)) * " "
            end
          elseif ((r - 1) % 3) == 2
            # South pulp cells of NW board cell => print legal actions code or centrality
            if ((c - 1) % 3) == 1
              # SW pulp cell of NW board cell => print 1st char of legal actions code (fusions) or centrality 1st digit
              if botmargin
                print_mark = " " * string(centrality ÷ 10)
              else
                  first_action = (column_row_to_index((column, row)) - 1) * NUM_ACTIONS_PER_CELL + 1
                  print_mark = " " * string(booleans_to_integer(amask[first_action + 4:first_action + 7]), base=16)
              end
            else
              # SE pulp cell of NW board cell => print 2nd char of legal actions code (divisions) or centrality 2nd digit
              if botmargin
                print_mark = string(centrality % 10) * " "
              else
                first_action = (column_row_to_index((column, row)) - 1) * NUM_ACTIONS_PER_CELL + 1
                print_mark = string(booleans_to_integer(amask[first_action:first_action + 3]), base=16) * " "
              end
            end
          end # Pulp cell of NW corner has its print_mark
        else
          # Pulp cell of non-NW corner
          print_mark = "  "
        end # Pulp cells have their print_mark
      end # All cells have their print_mark
      print(cell_color(print_value), print_mark, crayon"reset")
    end # Row is completely printed
    if (r - 2) % 3 == 1
      # Print row label
      print(" ", string(row, base=16))
    end
    print("\n")
  end
  # Print column labels
  for c in 0:BOARD_SIDE - 1
    print("    ", string(c, base=16))
  end
  print("\n")
  print("\n")
end
#
function read_row!(row_index::Int, input::String, board::Array, impact::Array, actions_hook::Array)
  s = input
  l = length(s)
  if s[l] == " "
    s = s[1:l - 1]
  end
  cells = split(s, " ")
  for column_index in 1:BOARD_SIDE
    values = split(cells[column_index], ".")
    board[row_index, column_index] = parse(Int, values[1], base=16)
    impact[row_index, column_index] = (parse(Int, values[3], base=10), parse(Int, values[4], base=10))
    column = parse(Int, values[2][1], base=16)
    row = parse(Int, values[2][2], base=16)
    actions_hook[row_index, column_index] = (column, row)
  end
end
#
function read_game(::GameSpec)
  g = GI.init(GameSpec)
  GI.set_state!(g, read_state(GameSpec))
  return g
end
#
function dump_game(g::GameEnv)
  for row_index in 1:BOARD_SIDE
    for column_index in 1:BOARD_SIDE
      cell_value = g.board[row_index, column_index]
      impact = g.impact[row_index, column_index]
      actions_hook = g.actions_hook[row_index, column_index]
      print(string(cell_value, base=16, pad=2), ".",
        string(actions_hook[1], base=16), string(actions_hook[2], base=16), ".",
        string(impact[1], base=10, pad=2), ".", string(impact[2], base=10, pad=2),  " ")
    end
    print("\n")
  end
  print(player_name(g.curplayer))
end
#
function read_state_array(rows::Array)
  board = Array(ZERO_BOARD)
  impact = Array(ZERO_TUPLE)
  actions_hook = Array(EMPTY_TUPLE)
  try
    for row_index in 1:BOARD_SIDE
      input = rows[row_index]
      read_row!(row_index, input, board, impact, actions_hook)
    end
    curplayer = rows[BOARD_SIDE + 1] == player_name(BLACK) ? BLACK : WHITE
    return (board=Board(board), impact=BoardTuple(impact), actions_hook=BoardTuple(actions_hook), curplayer=curplayer)
  catch e
    return nothing
  end
end
#
function GI.read_state(::GameSpec)
  board = Array(ZERO_BOARD)
  impact = Array(ZERO_TUPLE)
  actions_hook = Array(EMPTY_TUPLE)
  try
    for row_index in 1:BOARD_SIDE
      read_row!(row_index, readline(), board, impact, actions_hook)
    end
    curplayer = readline() == player_name(BLACK) ? BLACK : WHITE
    return (board=board, impact=impact, actions_hook=actions_hook, curplayer=curplayer)
  catch e
    return nothing
  end
end
