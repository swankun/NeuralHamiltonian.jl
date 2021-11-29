export ParametricPolicy, NeuralPBCPolicy, LinearPolicy

abstract type ParametricPolicy end

struct NeuralPBCPolicy{TP,T} <: ParametricPolicy
    prob::TP
    umax::T
end

function NeuralPBCPolicy(prob::NeuralPBCProblem, umax)
    T = precisionof(prob)
    NeuralPBCPolicy{typeof(prob),T}(prob,T(umax))
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
    umax::T
end
function (policy::LinearPolicy)(x,ps=nothing)
    @assert length(x) === policy.N
    K = policy.gains
    effort = -dot(K, x) 
    return clamp(effort, -policy.umax, policy.umax)
end

