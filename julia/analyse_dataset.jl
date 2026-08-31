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

ntot_kg = [sum(c.mode_N) for c in cases]
# Keep aerosol volume-concentration diagnostics referenced to saturated air,
# as before. This is independent of the actual RH=0.95 parcel initial state.
rhod0 = [dry_air_density(c.P0, c.T0, 1.0) for c in cases]
ntot_m3 = ntot_kg .* rhod0
nmodes = [length(c.mode_N) for c in cases]
smax = [maximum(c.S) for c in cases]
ndmax = [maximum(c.Nd) for c in cases]
ndfinal = [c.Nd[end] for c in cases]
ndkgmax = [maximum(c.Nd_kg) for c in cases]

# Exact activation fraction: both quantities are native number mixing ratios
# per kg dry air, so changes in air density with height cancel out.
fact_profiles = [[c.Nd_kg[j]/max(ntot_kg[i],1.0) for j in eachindex(c.height)] for (i,c) in enumerate(cases)]
fact = [maximum(f) for f in fact_profiles]
fact0 = [f[1] for f in fact_profiles]
s0 = [c.S[1] for c in cases]
z_smax = [c.height[argmax(c.S)] for c in cases]
z_fact90 = Float64[]
for (i,c) in enumerate(cases)
    f = fact_profiles[i]
    if fact[i] <= 1e-12
        push!(z_fact90, NaN)
    else
        j = findfirst(x -> x >= 0.9*fact[i], f)
        push!(z_fact90, j === nothing ? NaN : c.height[j])
    end
end
bmax = [maximum(c.beta_ext) for c in cases]
qlmax = [maximum(c.ql) for c in cases]
de50 = [maximum(filter(isfinite,c.deff); init=0f0) for c in cases]

# Total-water conservation diagnostic for the present adiabatic,
# non-entraining, liquid-only BMM truth trajectories:
#   qtot = ql + (1+S) * eps * es(T)/(P-es(T)).
qt_profiles = [total_water_profile(c) for c in cases]
qt0 = [qt[1] for qt in qt_profiles]
qt_final_drift = [qt[end] - qt[1] for qt in qt_profiles]
qt_max_abs_drift = [maximum(abs.(qt .- qt[1])) for qt in qt_profiles]
qt_max_rel_drift = [maximum(abs.(qt .- qt[1])) / max(abs(qt[1]), 1e-12) for qt in qt_profiles]

println("Dataset: $path")
println("successful trajectories: $(length(cases))")
println("mode-count distribution:")
for n in sort(unique(nmodes)); println("  $n modes: $(count(==(n),nmodes))"); end
println()
qline("updraft",[c.w for c in cases];unit="m s^-1")
qline("initial T",[c.T0 for c in cases];unit="K")
qline("initial P",[c.P0 for c in cases];scale=0.01,unit="hPa")
qline("total aerosol N",ntot_m3;scale=1e-6,unit="cm^-3 at saturated reference")
qline("total aerosol mixing",ntot_kg;unit="kg^-1 dry air")
qline("Smax",smax;scale=100,unit="%")
qline("Nd max",ndmax;scale=1e-6,unit="cm^-3")
qline("Nd final",ndfinal;scale=1e-6,unit="cm^-3")
qline("activated fraction",fact;unit="")
qline("initial |S|",abs.(s0);scale=100,unit="%")
qline("initial activation",fact0;unit="")
qline("height of Smax",z_smax;unit="m")
finite_z90 = filter(isfinite,z_fact90)
isempty(finite_z90) || qline("height of 90% act.",finite_z90;unit="m")
qline("max extinction",bmax;unit="m^-1")
qline("max ql",qlmax;unit="kg kg^-1")
qline("max deff",de50;scale=1e6,unit="um")

println("\nTotal-water conservation (qtot = ql + (1+S) qvs):")
qline("initial qtot",qt0;scale=1e3,unit="g kg^-1")
qline("final qtot drift",qt_final_drift;scale=1e6,unit="mg kg^-1")
qline("max |qtot drift|",qt_max_abs_drift;scale=1e6,unit="mg kg^-1")
qline("max relative drift",qt_max_rel_drift;scale=100,unit="%")
worst_qt = argmax(qt_max_rel_drift)
@printf("  worst case: %d  max |dqtot| = %.6g mg kg^-1  max relative drift = %.6g %%\n",
        cases[worst_qt].id, 1e6*qt_max_abs_drift[worst_qt], 100*qt_max_rel_drift[worst_qt])

println("\nInitial-state boundary checks used by surrogate v2:")
@printf("  S(0): min = %.6g %%  max = %.6g %%\n",100*minimum(s0),100*maximum(s0))
@printf("  max f_act(0) = %.6g\n",maximum(fact0))
println("\nActivation-regime counts (using max Nd_kg / total aerosol number mixing ratio):")
for (label,lo,hi) in (("<1%",-Inf,0.01),("1-10%",0.01,0.10),("10-50%",0.10,0.50),
                      ("50-90%",0.50,0.90),("90-110%",0.90,1.10),(">110%",1.10,Inf))
    n=count(x -> lo <= x < hi, fact)
    @printf("  %-8s %4d  (%5.1f%%)\n",label,n,100n/length(fact))
end

summary_path = splitext(path)[1] * "_case_summary.csv"
open(summary_path,"w") do io
    println(io,"case,n_modes,w,T0,P0,rhod_sat_ref,N_total_kg-1,N_total_sat_ref_m-3,S0,Smax,Nd_max_m-3,Nd_max_kg-1,Nd_final_m-3,activated_fraction,beta_ext_max,ql_max,deff_max,qtot0_kgkg,qtot_final_drift_kgkg,qtot_max_abs_drift_kgkg,qtot_max_rel_drift")
    for (k,c) in enumerate(cases)
        println(io,join((c.id,length(c.mode_N),c.w,c.T0,c.P0,rhod0[k],ntot_kg[k],ntot_m3[k],s0[k],smax[k],ndmax[k],ndkgmax[k],ndfinal[k],fact[k],bmax[k],qlmax[k],de50[k],qt0[k],qt_final_drift[k],qt_max_abs_drift[k],qt_max_rel_drift[k]),","))
    end
end
println("\nwrote $summary_path")
