export SetDistanceLoss, gradient

abstract type NeuralPBCLoss end

struct SetDistanceLoss{T,F} <: NeuralPBCLoss
    xstar::Vector{T}
    radius::T
    f::F
end
function SetDistanceLoss(f::Function,xstar,r)
    SetDistanceLoss{eltype(xstar),typeof(f)}(xstar,r,f)
end
function (l::SetDistanceLoss)(x::Matrix)
    # delta = minimum(map(l.f, eachcol(x)))
    # return delta < l.radius ? zero(eltype(x)) : delta - l.radius
    sum(abs2, x)
end

function gradient(l::NeuralPBCLoss, rollout::TrajectoryRollout, x0, θ)
    loss(ps) = l( Array(rollout(x0,ps)) )
    Zygote.gradient(loss, θ)[1]
end

function gradient(ls::Tuple{Vararg{L}}, rollout::TrajectoryRollout, x0, θ) where {L<:NeuralPBCLoss}
    loss(ps) = begin
        x = Array(rollout(x0,ps))
        sum(l(x) for l in ls)
    end
    Zygote.gradient(loss, θ)[1]
end
