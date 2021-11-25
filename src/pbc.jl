export NeuralPBCProblem, controller, predict

struct NeuralPBCProblem{iip,T,HD,F1}
    N::Int
    ham::HD
    dynamics::F1    # Closure dynamics(policy(x,θ)) -> f!(dx, x, ps, t)
    θ::Vector{T}
    ps_index::Dict{Symbol,UnitRange{Int}}
end

function NeuralPBCProblem{iip}(N, ham, dynamics) where {iip}
    !isa(ham, NeuralNetwork) && error("Not supported")
    netθ = ham.θ
    θ = [netθ; randn(N)]
    ps_index = Dict(
        :net => 1 : length(netθ), 
        :gains => length(netθ)+1 : length(netθ)+N
    )

    NeuralPBCProblem{iip,eltype(θ),typeof(ham),typeof(dynamics)}(
        N, ham, dynamics, θ, ps_index);
end
NeuralPBCProblem(ham, dynamics) = NeuralPBCProblem{false}(ham, dynamics)

function Base.show(io::IO, p::NeuralPBCProblem{iip}) where {iip}
    print(io, "$(p.N)-dimensional NeuralPBCProblem. In-place=$(iip)")
end

function Base.getproperty(p::NeuralPBCProblem, sym::Symbol)
    if sym === :θN
        return getindex(p.θ, p.ps_index[:net])
    elseif sym === :K
        return getindex(p.θ, p.ps_index[:gains])
    else # fallback to getfield
        return getfield(p, sym)
    end
end

function controller(p::NeuralPBCProblem, umax=Inf)
    Hd = p.ham
    u(x, ps) = begin
        θ = getindex(ps, p.ps_index[:net])
        K = getindex(ps, p.ps_index[:gains])
        effort = dot(K, gradient(Hd, x, θ)) 
        return clamp(effort, -umax, umax)
    end
end
