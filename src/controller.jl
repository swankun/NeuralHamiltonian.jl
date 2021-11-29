export ParametricPolicy, NeuralPBCPolicy

abstract type ParametricPolicy end

struct NeuralPBCPolicy{TP,T} <: ParametricPolicy
    prob::TP
    xstar::Vector{T}
    umax::T
end

function NeuralPBCPolicy(prob::NeuralPBCProblem, umax, xstar)
    NeuralPBCPolicy{typeof(prob),eltype(xstar)}(prob,xstar,umax)
end
function (pbc::NeuralPBCPolicy)(x,ps)
    θ = getindex(ps, pbc.prob.ps_index[:net])
    K = getindex(ps, pbc.prob.ps_index[:gains])
    effort = dot(K, gradient(pbc.prob.ham, x, θ)) 
    return clamp(effort, -pbc.umax, pbc.umax)
end

struct LinearPolicy{T} <: ParametricPolicy
    N::Int
    gains::Vector{T}
end

struct SwitchingPolicy{SW,ST} <: ParametricPolicy
    swing_policy::SW
    lin_policy::ST
end
