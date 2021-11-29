module NeuralHamiltonian

using LinearAlgebra
using Random
using Statistics

using Revise
using OrdinaryDiffEq, DiffEqSensitivity

import MLBasedESC: ReverseDiff, Zygote, NeuralNetwork, gradient

include("pbc.jl")
include("controller.jl")
include("rollout.jl")
include("loss.jl")
include("systems/abstractsys.jl")
include("systems/acrobot.jl")

end # module
