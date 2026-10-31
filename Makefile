# Julia parcel model / BMM surrogate
#
# bmm/ is a git subtree from the separate Bin Microphysics Model repository.
# This wrapper invokes it but does not modify its Makefiles or source layout.

BMM_DIR ?= bmm
JULIA ?= julia

CASES ?= 256
MAX_MODES ?= 4
WORKERS ?= 64
SURROGATE_SEED ?= 20260831
PROFILE_EPOCHS ?= 150
NEURALODE_EPOCHS ?= 40
NPLOTS ?= 8
DEVICE ?= auto

DATA_DIR ?= data
MODEL_DIR ?= models
RESULTS_DIR ?= results
BMM_DATASET ?= $(DATA_DIR)/synthetic_bmm.nc

UNAME_S := $(shell uname -s)

# The BMM subtree's defaults work on macOS.  On Ubuntu the following command-
# line overrides are required.  Passing them here leaves bmm/ untouched.
ifeq ($(UNAME_S),Linux)
BMM_BUILD_ARGS = \
	NETCDFLIB=-L/usr/lib/x86_64-linux-gnu/ \
	NETCDFMOD=/usr/include/ \
	FFLAGS='-O2 -w -o' \
	FFLAGS2='-O2 -w -o' \
	FFLAGSOMP='-O2 -w -o'
else
BMM_BUILD_ARGS =
endif

.PHONY: help setup build bmm clean clean-bmm test julia-instantiate julia-test \
        smoke generate synthetic-pilot analyse analyse-dataset \
        train train-profile train-neuralode compare compare-surrogates dirs \
        gpu-setup gpu-check

help:
	@echo ""
	@echo "BMM-trained Julia cloud-activation surrogate"
	@echo ""
	@echo "Quick start:"
	@echo "  make setup                 Instantiate the Julia environment"
	@echo "  make build                 Build BMM (macOS/Linux detected automatically)"
	@echo "  make smoke                 Test Julia -> BMM -> NetCDF -> Julia"
	@echo ""
	@echo "Main workflow:"
	@echo "  make generate              Generate synthetic BMM training data"
	@echo "  make analyse               Summarise/check the generated dataset"
	@echo "  make train                 Train (GPU if available, otherwise CPU)"
	@echo "  make train DEVICE=gpu      Train profile surrogate on NVIDIA GPU"
	@echo "  make compare               Evaluate and plot held-out cases"
	@echo ""
	@echo "Useful overrides:"
	@echo "  CASES=256                  Number of synthetic BMM cases"
	@echo "  WORKERS=64                 Maximum concurrent BMM processes"
	@echo "  MAX_MODES=4                Maximum aerosol modes per case"
	@echo "  SURROGATE_SEED=20260831    Reproducible sampling/train split seed"
	@echo "  PROFILE_EPOCHS=150         Profile-network training epochs"
	@echo "  DEVICE=auto|cpu|gpu        Training device (default: auto)"
	@echo "  BMM_DATASET=data/file.nc   Dataset path from repository root"
	@echo ""
	@echo "Example:"
	@echo "  make generate CASES=2000 WORKERS=128"
	@echo "  make train DEVICE=gpu"
	@echo ""
	@echo "Detected platform: $(UNAME_S)"
	@echo ""

dirs:
	@mkdir -p $(DATA_DIR) $(MODEL_DIR) $(RESULTS_DIR) scratch

setup: dirs julia-instantiate
build: bmm
test: julia-test
generate: synthetic-pilot
analyse: analyse-dataset
train: train-profile
compare: compare-surrogates
clean: clean-bmm

bmm:
	$(MAKE) -C $(BMM_DIR) $(BMM_BUILD_ARGS)

clean-bmm:
	$(MAKE) -C $(BMM_DIR) cleanall

julia-instantiate:
	cd julia && $(JULIA) --project=. -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()'

# Optional NVIDIA training dependencies. This intentionally remains separate
# from `make setup` so the same base environment remains convenient on macOS.
gpu-setup:
	cd julia && $(JULIA) --project=. -e 'using Pkg; Pkg.add(["CUDA", "cuDNN"]); Pkg.precompile()'

gpu-check:
	cd julia && $(JULIA) --project=. -e 'using CUDA; println("CUDA functional = ", CUDA.functional()); CUDA.functional() && CUDA.versioninfo()'

julia-test:
	cd julia && $(JULIA) --project=. test/runtests.jl

smoke: bmm julia-instantiate
	cd julia && BMM_EXE=../$(BMM_DIR)/main.exe $(JULIA) --project=. smoke.jl

synthetic-pilot: bmm julia-instantiate dirs
	cd julia && \
	BMM_EXE=../$(BMM_DIR)/main.exe \
	BMM_WORKERS=$(WORKERS) \
	$(JULIA) --project=. -t $(WORKERS) generate_synthetic_dataset.jl \
	$(CASES) ../$(BMM_DATASET) $(MAX_MODES) $(SURROGATE_SEED) $(WORKERS)

analyse-dataset: julia-instantiate
	cd julia && $(JULIA) --project=. analyse_dataset.jl ../$(BMM_DATASET)

train-profile: julia-instantiate dirs
	cd julia && \
	MODEL_DIR=../$(MODEL_DIR) RESULTS_DIR=../$(RESULTS_DIR) DEVICE=$(DEVICE) \
	$(JULIA) --project=. train_surrogate.jl \
	../$(BMM_DATASET) profile $(PROFILE_EPOCHS) $(SURROGATE_SEED)

train-neuralode: julia-instantiate dirs
	cd julia && \
	MODEL_DIR=../$(MODEL_DIR) RESULTS_DIR=../$(RESULTS_DIR) DEVICE=$(DEVICE) \
	$(JULIA) --project=. train_surrogate.jl \
	../$(BMM_DATASET) neuralode $(NEURALODE_EPOCHS) $(SURROGATE_SEED)

compare-surrogates: julia-instantiate dirs
	cd julia && \
	MODEL_DIR=../$(MODEL_DIR) RESULTS_DIR=../$(RESULTS_DIR) \
	$(JULIA) --project=. compare_surrogates.jl \
	../$(BMM_DATASET) $(SURROGATE_SEED) profile $(NPLOTS)
