using Test
include(joinpath(@__DIR__, "..", "BMM.jl"))
include(joinpath(@__DIR__, "..", "SyntheticDataset.jl"))
using .BMM
using .SyntheticDataset

@testset "cloud-base geometry" begin
    c = cloud_base_case(winit=2.0, height_top=500.0, max_dt=5.0, max_dz=4.0)
    @test c.runtime ≈ 250.0
    @test c.dt ≈ 2.0
    @test c.rhinit <= 1.0
end

@testset "ideal Koehler / kappa equivalence mapping" begin
    m = AerosolMode(nu=3.0, molw=132.14e-3, density=1770.0)
    @test effective_kappa(m) > 0
end

@testset "multimode namelist" begin
    c = cloud_base_case(winit=1.0, height_top=100.0,
        modes=[AerosolMode(N=1e8, Dm=30e-9, kappa=0.1),
               AerosolMode(N=2e8, Dm=150e-9, kappa=0.7)])
    s = to_namelist(c, "/tmp/test.nc")
    @test occursin("n_mode=2", s)
    @test occursin("n_intern=1", s)
    @test occursin("n_comps=2", s)
    @test occursin("mass_frac_aer1(1,1)=1.0", s)
    @test occursin("mass_frac_aer1(1,2)=0.0", s)
    @test occursin("mass_frac_aer1(2,2)=1.0", s)
end

@testset "Dcrit diagnostic" begin
    nwat = [1.0, 1.0, 1.0, 1.0]
    nliq = [0.0, 0.1, 0.8, 1.0]
    edges = [1e-21, 2e-21, 4e-21, 8e-21, 16e-21]
    d = derive_dcrit(nliq, nwat, edges, 1800.0)
    @test isfinite(d)
    @test d > 0
    @test isnan(derive_dcrit(zeros(4), nwat, edges, 1800.0))
end

@testset "synthetic sampling" begin
    cfg = SamplingConfig(n_cases=12, max_modes=4, seed=1)
    cases, classes = sample_cases(cfg)
    @test length(cases) == 12
    @test sort(unique(length.(getfield.(cases, :modes)))) == [1, 2, 3, 4]
    @test length(classes) == 12
    @test all(c -> 0.1 <= c.winit <= 10.0, cases)
    @test all(c -> all(m -> 0.0 <= m.kappa <= 1.4, c.modes), cases)
end

@testset "fixed BMM numerical scheme" begin
    c = cloud_base_case(winit=1.0, height_top=20.0)
    nml = BMM.to_namelist(c, "dummy.nc")
    @test occursin("bin_scheme_flag=0", nml)
    @test occursin("sce_flag=0", nml)
    @test !hasfield(BMMCase, :bin_scheme_flag)
    @test !hasfield(BMMCase, :sce_flag)
end

@testset "fixed liquid-only truth configuration" begin
    nml = to_namelist(cloud_base_case(), "test.nc")
    @test occursin("ice_flag=0", nml)
    @test occursin("bin_scheme_flag=0", nml)
    @test occursin("sce_flag=0", nml)
end

# Surrogate v2 regression tests. Keep these after the light BMM-interface tests
# so failures in the ML representation are reported separately.
include(joinpath(@__DIR__, "..", "SurrogateData.jl"))
include(joinpath(@__DIR__, "..", "SetSurrogate.jl"))
using .SurrogateData
using .SetSurrogate
using Flux

function _mlcase(mode_N, mode_Dm, mode_lnsig, mode_kappa)
    nm = length(mode_N)
    h = Float32[0, 5, 10]
    ntot = sum(mode_N)
    MLCase(
        1, h,
        Float32.(mode_N), Float32.(mode_Dm), Float32.(mode_lnsig), Float32.(mode_kappa),
        fill(3f0, nm), fill(0.13214f0, nm), fill(1770f0, nm),
        280f0, 90000f0, 1f0, fill(1.0f0, length(h)),
        Float32[0, 0.002, 0.001],
        Float32[0, 0.2, 0.2] .* Float32(ntot),
        Float32[0, 0.2, 0.2] .* Float32(ntot),
        Float32[1e-4, 2e-4, 3e-4],
        Float32[0, 1e-4, 2e-4],
        Float32[0, 10e-6, 15e-6],
        fill(Float32(NaN), length(h), nm)
    )
end

@testset "surrogate v2 activation transform" begin
    for f in (0.0, 1e-5, 0.01, 0.1, 0.5, 0.999, 1.0)
        @test latent_fraction(fraction_latent(f)) ≈ f atol=2e-5
    end
    @test 0.0 <= latent_fraction(-100.0) <= 1.0
    @test 0.0 <= latent_fraction(100.0) <= 1.0
    @test critical_supersaturation(200e-9, 0.6, 280.0) <
          critical_supersaturation(50e-9, 0.6, 280.0)
end

@testset "set encoder invariances" begin
    one = _mlcase([2e8], [100e-9], [0.5], [0.4])
    split = _mlcase([1e8, 1e8], [100e-9, 100e-9], [0.5, 0.5], [0.4, 0.4])
    encoder = Chain(Dense(4, 8, tanh), Dense(8, 5, tanh))
    @test size(SetSurrogate._mode_features(one)) == (4, 1)
    @test size(SetSurrogate._mode_features(split)) == (4, 2)
    @test aerosol_context(encoder, one) ≈ aerosol_context(encoder, split) atol=2e-6

    a = _mlcase([1e8, 3e8], [50e-9, 200e-9], [0.4, 0.6], [0.2, 0.8])
    b = _mlcase([3e8, 1e8], [200e-9, 50e-9], [0.6, 0.4], [0.8, 0.2])
    @test aerosol_context(encoder, a) ≈ aerosol_context(encoder, b) atol=2e-6
end

@testset "profile hard physical constraints" begin
    c = _mlcase([2e8], [100e-9], [0.5], [0.4])
    p, m = build_profile_model(embed_dim=5, hidden=8, hscale=10.0)
    pred = predict_profile(m, p, c)
    @test pred.S[1] == 0.0
    @test pred.activation_fraction[1] == 0.0
    @test pred.Nd_kg[1] == 0.0
    @test all(f -> 0.0 <= f <= 1.0, pred.activation_fraction)
end

@testset "BMM number diagnostic units" begin
    c = cloud_base_case(modes=[AerosolMode(N=1e8, Dm=100e-9)])
    txt = case_diagnostics(c)
    @test occursin("kg^-1 dry air", txt)
    @test occursin("cm^-3 at cloud base", txt)
end
