include("BMM.jl")
using .BMM

exe = get(ENV, "BMM_EXE", BMM.default_bmm_exe())
modes = [
    AerosolMode(N=500e6, Dm=55e-9, lnsig=0.45, kappa=0.25),
    AerosolMode(N=150e6, Dm=180e-9, lnsig=0.55, kappa=0.65),
]
case = cloud_base_case(winit=1.0, height_top=100.0, max_dt=2.0, max_dz=2.0,
                       tinit=283.0, pinit=90000.0, modes=modes)
r = run_bmm(case; exe=exe)
q = resample_profile(r; dz=5.0, height_top=100.0)

@assert length(q.height) > 2
@assert all(diff(q.height) .> 0)
@assert all(isfinite, q.ndrop)
@assert all(isfinite, q.ndrop_kg)
@assert all(isfinite, q.rhod)
@assert all(isfinite, q.beta_ext)
@assert size(q.dcrit, 2) == 2

println("BMM/Julia multimode smoke test OK")
println("height=$(q.height[end]) m Nd=$(q.ndrop[end]) m^-3 beta_ext=$(q.beta_ext[end]) m^-1")
