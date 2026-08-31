include("SurrogateData.jl")
include("SetSurrogate.jl")
using .SurrogateData
using .SetSurrogate

path=length(ARGS)>=1 ? ARGS[1] : "synthetic_bmm.nc"
seed=length(ARGS)>=2 ? parse(Int,ARGS[2]) : 20260831
cases=read_cases(path); split=split_cases(cases;seed=seed)

if isfile("profile_surrogate.bson")
    p,m=load_profile_model("profile_surrogate.bson")
    println("profile validation: ",evaluate_profile(m,p,split.val))
    println("profile TEST:       ",evaluate_profile(m,p,split.test))
else
    println("profile_surrogate.bson not found")
end
if isfile("neuralode_surrogate.bson")
    p,m=load_neuralode_model("neuralode_surrogate.bson")
    println("neuralODE validation: ",evaluate_neuralode(m,p,split.val))
    println("neuralODE TEST:       ",evaluate_neuralode(m,p,split.test))
else
    println("neuralode_surrogate.bson not found")
end
