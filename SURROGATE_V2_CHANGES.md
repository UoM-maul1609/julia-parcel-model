# Surrogate v2 refinement after the 256-case pilot

The first held-out profile comparison exposed three important weaknesses in the
initial surrogate formulation:

1. absolute `Nd` predictions were not constrained by available aerosol number;
2. near-zero activation was weakly represented by `log1p(Nd/1e8)`;
3. a pure pointwise loss smoothed the narrow supersaturation/activation peak.

The v2 implementation therefore makes the following changes.

## Physical target

The learned drop-number variable is now

`f_act(z) = Nd_kg(z) / sum(mode_N)`.

It is represented with a finite scaled-logit transform and inverted with an
explicit `[0,1]` bound. `Nd_kg` is reconstructed from `f_act * N_total`.
Consequently a surrogate prediction cannot exceed the available aerosol number.

The known cloud-base conditions `S(0)=0` and `f_act(0)=0` are imposed exactly.
Cloud-base extinction remains learned because hydrated haze can have finite
extinction before activation.

## Inputs and architecture

The set encoder remains permutation invariant and exactly invariant to splitting
an identical mode. Per-mode features are now:

- log dry median diameter;
- lognormal width;
- transformed effective kappa;
- approximate kappa-Koehler critical supersaturation at the median diameter.

The pooled aerosol context also contains total number, cloud-base T/P, and
lognormal M2/M3 bulk moments (dry surface/volume proxies). The direct profile
model has additional near-cloud-base height coordinates and modestly increased
capacity.

## Loss and optimization

The pointwise transformed-state MSE is supplemented by weak peak and final-state
terms. Training now uses gradient clipping, deterministic Flux initialization,
best-validation restoration, and patience-based early stopping.

## Diagnostics

`compare_surrogates.jl` now plots BMM and surrogate profiles of S, activated
fraction, Nd and extinction, caches Neural ODE predictions, writes aerosol
metadata for selected cases, produces peak 1:1 scatters, and reports errors by
mode count and activation regime.

Future synthetic NetCDF files also store P/T/RH/dry-density profiles. Existing
correct-unit 256-case datasets remain readable; no BMM rerun is required to test
surrogate v2, although old datasets use cloud-base density as the fallback unit
conversion for predicted Nd in m^-3.

## Compatibility

Surrogate v1 `.bson` files are intentionally rejected by the v2 loader because
the feature and target dimensions changed. Retrain them from the existing
NetCDF dataset.
