module BonbonRectangle
export GameEnv, GameSpec, Board
include("game.jl")
  module Training
    using AlphaZero
    import ..GameSpec
    include("params.jl")
  end
end
