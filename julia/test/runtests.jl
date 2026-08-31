using Test
include(joinpath(@__DIR__, "..", "BMM.jl"))
include(joinpath(@__DIR__, "..", "SyntheticDataset.jl"))
using .BMM
using .SyntheticDataset

@testset "cloud-base geometry" begin
    c = cloud_base_case(winit=2.0, height_top=500.0, max_dt=5.0, max_dz=4.0)
    @test c.runtime ≈ 250.0
    @test c.dt ≈ 2.0
    @test c.rhinit == 1.0
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
