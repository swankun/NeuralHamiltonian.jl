using MLBasedESC
import MLBasedESC: Flux
using NeuralHamiltonian
using OrdinaryDiffEq

function dynamics(policy::Function)
    g   = 9.81
    m1  = 2.7012465523580262
    m2  = 0.405
    lc1 = 0.18645114033013516
    lc2 = 0.177
    l1  = 0.6
    l2  = 0.532
    I1  = 2.7012465523580262*0.18645114033013516^2 + 0.209*1.1
    I2  = 0.405*0.177^2 + 0.01497763008
    b1  = 0.001
    b2  = 0.005
    ϵ = 0.05
    function f!(dx, x, ps, t)
        cq1, sq1, q1dot, cq2, sq2, q2dot = x
        sq1q2 = sq1*cq2 + cq1*sq2
        effort = policy(x,ps)
        dx[1] = -sq1*q1dot - ϵ*cq1*(sq1^2 + cq1^2 - 1)
        dx[2] = cq1*q1dot - ϵ*sq1*(sq1^2 + cq1^2 - 1)
        dx[3] = (-I2*(b1*q1dot + g*lc1*m1*sq1 + g*m2*(l1*sq1 + lc2*sq1q2) - 2*l1*lc2*m2*q1dot*q2dot*sq2 - l1*lc2*m2*q2dot^2*sq2) + (I2 + l1*lc2*m2*cq2)*(b2*q2dot + g*lc2*m2*sq1q2 + l1*lc2*m2*q1dot^2*sq2 - effort))/(I1*I2 + I2*l1^2*m2 - l1^2*lc2^2*m2^2*cq2^2)
        dx[4] = -sq2*q2dot - ϵ*cq2*(sq2^2 + cq2^2 - 1)
        dx[5] = cq2*q2dot - ϵ*sq2*(sq2^2 + cq2^2 - 1)
        dx[6] = ((I2 + l1*lc2*m2*cq2)*(b1*q1dot + g*lc1*m1*sq1 + g*m2*(l1*sq1 + lc2*sq1q2) - 2*l1*lc2*m2*q1dot*q2dot*sq2 - l1*lc2*m2*q2dot^2*sq2) - (I1 + I2 + l1^2*m2 + 2*l1*lc2*m2*cq2)*(b2*q2dot + g*lc2*m2*sq1q2 + l1*lc2*m2*q1dot^2*sq2 - effort))/(I1*I2 + I2*l1^2*m2 - l1^2*lc2^2*m2^2*cq2^2)
        nothing
    end
end

function inmap(x)
    vcat(cos(x[1]), sin(x[1]), x[3], cos(x[2]), sin(x[2]), x[4])
end
function outmap(x::Vector)
    vcat(atan(x[2], x[1]), atan(x[5], x[4]), x[3], x[6])
end
function outmap(x::Matrix)
    q1 = map(y->atan(y[2],y[1]), eachcol(x))
    q2 = map(y->atan(y[5],y[4]), eachcol(x))
    q1dot = view(x, 3, :)
    q2dot = view(x, 6, :)
    q1revs = cumsum( -round.(Int, [0; diff(q1)]/pi) )
    q2revs = cumsum( -round.(Int, [0; diff(q2)]/pi) )
    q1 = q1 .+ q1revs*pi
    q2 = q2 .+ q2revs*pi
    return [q1 q2 q1dot q2dot] |> transpose
end
distance(x) = begin
    +(
        4*2(1+x[1]),
        2*2(1-x[4]),
        2*x[3]^2,
        1*x[4]^2
    )
end

## Problem setup
Hd = NeuralNetwork(Float64, [6,16,48,1])
pbc = NeuralPBCProblem{true}(6, Hd, dynamics);
predict = TrajectoryRollout(pbc, tf=3.0, umax=1.0);

## Rollout
x0 = inmap([3.,0,0,0])
# predict(pbc, x0, pbc.θ)

## Loss
xdesired = inmap(zeros(4))
radius = 0.01
l1 = SetDistanceLoss(distance, xdesired, radius)
# l1(rand(6), pbc.θ)
# gradient(l1, predict, x0, pbc.θ)

## Minibatch
function batch(x0s,ps)
    N = length(x0s)
    gs = [zeros(length(ps)) for _=1:N]
    ls = zeros(N)
    Threads.@threads for i=1:N
        lossgrad, lossval = gradient(l1,predict,x0s[i],ps)
        ls[i] = lossval
        gs[i][:] = lossgrad
    end
    sum(gs)/N, sum(ls)/N
end

## Sampler
function newsample()
    q1 = rand([-1,1])*pi + 0.5*( rand()-0.5 )
    q2 = 0.5*( rand()-0.5 )
    q1dot = 0*( rand()-0.5 )
    q2dot = 0*( rand()-0.5 )
    return inmap([q1,q2,q1dot,q2dot])
end

## Training loop
function train!(θ)
    optimizer = Flux.ADAM()
    epoch = 0
    while epoch < 1000
        x0s = [newsample() for _=1:8];
        gs, ls = batch(x0s,θ)
        Flux.Optimise.update!(optimizer,θ,gs)
        @info "Loss is $ls"
        epoch += 1
    end
end
# ps = deepcopy(pbc.θ);
# train!(ps)

evaluate = transpose ∘ outmap ∘ Array
# evaluate(predict(x0,ps,tf=10.0,dt=0.01))
