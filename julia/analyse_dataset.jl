include("SurrogateData.jl")
using .SurrogateData
using Statistics
using Printf

path = length(ARGS) >= 1 ? ARGS[1] : get(ENV,"BMM_DATASET","synthetic_bmm.nc")
cases = read_cases(path)
isempty(cases) && error("no successful cases in $path")

q(v,p) = quantile(Float64.(v),p)
function qline(name,v;scale=1.0,unit="")
    vv=Float64.(v).*scale
    @printf("%-24s min=%10.4g  p10=%10.4g  p50=%10.4g  p90=%10.4g  max=%10.4g %s\n",
            name,minimum(vv),q(vv,0.1),q(vv,0.5),q(vv,0.9),maximum(vv),unit)
end

ntot = [sum(c.mode_N) for c in cases]
nmodes = [length(c.mode_N) for c in cases]
smax = [maximum(c.S) for c in cases]
ndmax = [maximum(c.Nd) for c in cases]
ndfinal = [c.Nd[end] for c in cases]
fact = [maximum(c.Nd)/max(sum(c.mode_N),1f0) for c in cases]
bmax = [maximum(c.beta_ext) for c in cases]
qlmax = [maximum(c.ql) for c in cases]
de50 = [maximum(filter(isfinite,c.deff); init=0f0) for c in cases]

println("Dataset: $path")
println("successful trajectories: $(length(cases))")
println("mode-count distribution:")
for n in sort(unique(nmodes)); println("  $n modes: $(count(==(n),nmodes))"); end
println()
qline("updraft",[c.w for c in cases];unit="m s^-1")
qline("cloud-base T",[c.T0 for c in cases];unit="K")
qline("cloud-base P",[c.P0 for c in cases];scale=0.01,unit="hPa")
qline("total aerosol N",ntot;scale=1e-6,unit="cm^-3")
qline("Smax",smax;scale=100,unit="%")
qline("Nd max",ndmax;scale=1e-6,unit="cm^-3")
qline("Nd final",ndfinal;scale=1e-6,unit="cm^-3")
qline("activated fraction",fact;unit="")
qline("max extinction",bmax;unit="m^-1")
qline("max ql",qlmax;unit="kg kg^-1")
qline("max deff",de50;scale=1e6,unit="um")

println("\nActivation-regime counts (using max Nd / total aerosol N):")
for (label,lo,hi) in (("<1%",-Inf,0.01),("1-10%",0.01,0.10),("10-50%",0.10,0.50),
                      ("50-90%",0.50,0.90),("90-110%",0.90,1.10),(">110%",1.10,Inf))
    n=count(x -> lo <= x < hi, fact)
    @printf("  %-8s %4d  (%5.1f%%)\n",label,n,100n/length(fact))
end

summary_path = splitext(path)[1] * "_case_summary.csv"
open(summary_path,"w") do io
    println(io,"case,n_modes,w,Tcb,Pcb,N_total,Smax,Nd_max,Nd_final,activated_fraction,beta_ext_max,ql_max,deff_max")
    for (k,c) in enumerate(cases)
        println(io,join((c.id,length(c.mode_N),c.w,c.T0,c.P0,ntot[k],smax[k],ndmax[k],ndfinal[k],fact[k],bmax[k],qlmax[k],de50[k]),","))
    end
end
println("\nwrote $summary_path")
