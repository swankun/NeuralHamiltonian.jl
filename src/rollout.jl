export TrajectoryRollout

struct TrajectoryRollout{PBC,OP}
    pbc::PBC
    ode::OP
    horizon::Float64
end

function TrajectoryRollout(pbc::NeuralPBCProblem{iip,T}, policy::P; tf) where {iip,T,P<:ParametricPolicy}
    closedloop = pbc.dynamics(policy)
    tspan = (T(0.0), T(tf))
    ode = ODEProblem{iip}(closedloop, zeros(pbc.N), tspan, pbc.θ)
    TrajectoryRollout{typeof(pbc),typeof(ode)}(pbc, ode, tf)
end
function (l::TrajectoryRollout)(x0, θ; tf=l.horizon, dt=0.1)
    solve(remake(l.ode, u0=x0, tspan=(0.0,tf)), Tsit5(); p=θ, 
        rtol=1e-6, atol=1e-6,
        saveat=dt, 
        sensealg=InterpolatingAdjoint()
    )
end
