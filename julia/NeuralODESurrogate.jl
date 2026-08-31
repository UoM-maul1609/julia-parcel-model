module NeuralODESurrogate
using Flux, DifferentialEquations, SciMLBase, SciMLSensitivity, Zygote, Statistics, Random
using BSON: @save, @load
export CaseData, SurrogateModel, build_model, train!, predict, save_model, load_model

struct CaseData
    height::Vector{Float64}; w::Vector{Float64}; S::Vector{Float64}; Nd::Vector{Float64}
    beta_ext::Vector{Float64}; Dcrit::Vector{Float64}; context::Vector{Float64}
end
const H_SCALE=1000f0; const S_SCALE=5f-3; const ND_SCALE=1f8; const B_SCALE=1f-3; const W_FLOOR=1f-3
state(c,i)=Float32[c.S[i]/S_SCALE,log1p(max(c.Nd[i],0)/ND_SCALE),log1p(max(c.beta_ext[i],0)/B_SCALE)]
physical(y)=(S=Float64(y[1]*S_SCALE),Nd=max(0.0,Float64(expm1(y[2])*ND_SCALE)),beta_ext=max(0.0,Float64(expm1(y[3])*B_SCALE)))
struct SurrogateModel; re; ncontext::Int; hidden::Int; end
function build_model(;ncontext::Int,hidden::Int=64)
    net=Chain(Dense(ncontext+4,hidden,tanh),Dense(hidden,hidden,tanh),Dense(hidden,3)); p,re=Flux.destructure(net)
    p,SurrogateModel(re,ncontext,hidden)
end
function interp(x,xs,ys)
    x<=xs[1] && return ys[1]; x>=xs[end] && return ys[end]; i=searchsortedlast(xs,x)
    f=(x-xs[i])/(xs[i+1]-xs[i]); ys[i]+f*(ys[i+1]-ys[i])
end
function solve_case(m,p,c;solver=Tsit5(),saveat=c.height)
    hs=Float32.(c.height); ws=Float32.(c.w); ctx=Float32.(c.context)
    function rhs(u,p,hhat)
        h=hhat*H_SCALE; wf=log(max(interp(h,hs,ws),W_FLOOR)); m.re(p)(vcat(u,wf,ctx))
    end
    hh=Float32.(c.height./H_SCALE); prob=ODEProblem(rhs,state(c,1),(hh[1],hh[end]),p)
    solve(prob,solver;saveat=Float32.(saveat./H_SCALE),sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()),abstol=1f-6,reltol=1f-6)
end
function case_loss(m,p,c)
    sol=solve_case(m,p,c); SciMLBase.successful_retcode(sol) || return 1f6
    sum(sum(abs2,sol.u[i].-state(c,i)) for i in eachindex(c.height))/length(c.height)
end
function train!(p,m,cases;epochs=200,lr=1e-3,batchsize=8,val_cases=nothing,rng=Random.default_rng())
    opt=Flux.setup(Adam(lr),p); n=length(cases)
    for ep in 1:epochs
        perm=randperm(rng,n); total=0.0
        for s in 1:batchsize:n
            ix=perm[s:min(s+batchsize-1,n)]; batch=cases[ix]
            loss,g=Flux.withgradient(p) do theta; mean(case_loss(m,theta,c) for c in batch); end
            Flux.update!(opt,p,g[1]); total+=loss*length(ix)
        end
        if ep==1 || ep%10==0
            msg="epoch $ep train_loss=$(total/n)"; val_cases!==nothing && !isempty(val_cases) && (msg*=" val_loss=$(mean(case_loss(m,p,c) for c in val_cases))")
            println(msg)
        end
    end; p
end
function predict(m,p,c)
    sol=solve_case(m,p,c); vals=[physical(u) for u in sol.u]
    (height=Float64.(sol.t).*H_SCALE,S=[x.S for x in vals],Nd=[x.Nd for x in vals],beta_ext=[x.beta_ext for x in vals])
end
function save_model(p,m,path)
    ncontext=m.ncontext; hidden=m.hidden; @save path p ncontext hidden
end
function load_model(path)
    @load path p ncontext hidden; _,m=build_model(ncontext=ncontext,hidden=hidden); p,m
end
end
