import Pkg; Pkg.add("Optim")

using Optim

f(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2

a=optimize(f, [0.0, 0.0],Optim.Options(x_abstol=1e0,f_abstol=1e0))
b=Optim.minimum(a)
println(b)

# try another function

f(x) = x[1]^2-3*x[1]+2

a=optimize(f, [0.0],Optim.Options(x_abstol=1e0,f_abstol=1e0))
b=Optim.minimum(a) #fmin

c=Optim.minimizer(a) # xmin

println(b)
println(c)
