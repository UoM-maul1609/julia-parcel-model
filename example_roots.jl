import Pkg; Pkg.add("Roots")

using Roots

f(x) = exp(x) - x^4;

# α₀, α₁, α₂ = -0.8155534188089607, 1.4296118247255556, 8.6131694564414;


# bisection methods
a=find_zero(f, (8,9), Bisection()) # a bisection method has the bracket specified
true
println(a)

a=find_zero(f, (-10, 0)) # Bisection is default if x in `find_zero(f, x)` is not scalar
true
println(a)

a= find_zero(f, (-10, 0), Roots.A42()) # fewer function evaluations than Bisection
true
println(a)


# secant / non-bracketing methods

a=find_zero(f, 3) # find_zero(f, x0::Number) will use Order0()
true
println(a)

a=find_zero(f, -1) # find_zero(f, x0::Number) will use Order0()
true
println(a)

a=find_zero(f, 9) # find_zero(f, x0::Number) will use Order0()
true
println(a)

