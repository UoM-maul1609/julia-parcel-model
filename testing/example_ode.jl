using Pkg
# Pkg.add("DifferentialEquations")

using SciMLBase


function lorenz!(du,u,p,t)
        du[1] = 10.0(u[2]-u[1])
        du[2] = u[1]*(28.0-u[3]) - u[2]
        du[3] = u[1]*u[2] - (8/3)*u[3]
end

u0 = [1.0;0.0;0.0]

tspan = (0.0,100.0)

prob = ODEProblem(lorenz!,u0,tspan)

# Test that it worked
# https://docs.sciml.ai/DiffEqDocs/stable/solvers/ode_solve/#Translations-from-MATLAB/Python/R

using OrdinaryDiffEq


sol = solve(prob,Tsit5(),abstol=1.e-10, reltol=1.e-15)

# import Pkg; Pkg.add("Plots")
using Plots; 
#plot(sol,idxs=(1,2,3), c=:red)

# stiff method, like VODE
prob = ODEProblem(lorenz!,u0,tspan)

sol = solve(prob,QNDF(),abstol=1.e-10, reltol=1.e-15)
plot(sol,idxs=(1,2,3), c=:green)

# stiff method, like VODE
prob = ODEProblem(lorenz!,u0,tspan)

sol = solve(prob,FBDF(),abstol=1.e-10, reltol=1.e-15)
plot!(sol,idxs=(1,2,3), c=:blue)

# Pkg.add("LSODA")
using LSODA
# stiff method, like lsoda
prob = ODEProblem(lorenz!,u0,tspan)

sol = solve(prob,lsoda(),abstol=1.e-10, reltol=1.e-15)
plot!(sol,idxs=(1,2,3), c=:black)
