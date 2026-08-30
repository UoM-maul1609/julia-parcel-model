!> bmm_capi.f90
!>
!> Thin C-interoperable wrapper around the BMM parcel driver, for building a
!> shared library (libbmm.so) that can be ccall'd from Julia.
!>
!> *** IMPORTANT REENTRANCY CAVEAT ***
!> bin_microphysics_module.f90 allocates its module-level arrays on every call
!> to initialise_bmm_arrays/bmm_driver without freeing them first. Confirmed
!> experimentally: calling the driver twice in the same OS process fails on
!> the second call ("STOP *** Not enough memory ***"), because the runtime
!> refuses to re-ALLOCATE an already-allocated array.
!>
!> Consequences for how you use this library from Julia:
!>   - Safe:   load libbmm.so, call bmm_run_c ONCE, then exit the process.
!>   - Unsafe: calling bmm_run_c more than once from the same long-lived Julia
!>             session (e.g. in a training loop or ensemble sweep).
!>
!> For repeated calls (parameter sweeps, online UDE loss evaluation, etc.),
!> use the process-per-call interface in julia/BMM.jl instead (run_bmm, which
!> spawns main.exe fresh each time -- this is the *recommended default* and is
!> what's used by the rest of this project). It has ~0.25 s overhead per call
!> for a 3000 s parcel run, which is fine for offline data generation and is
!> trivially parallel across OS processes/cores.
!>
!> To make this library truly reentrant, the fix is to add explicit
!> `if (allocated(x)) deallocate(x)` guards before every `allocate(x(...))` in
!> bin_microphysics_module.f90 (and in sce_module.f90's fixed-grid setup, if
!> bin_scheme_flag > 0 is ever used). That is a real patch to the upstream
!> BMM source, not something a wrapper file can paper over, so it's out of
!> scope here -- flagging it so you can decide whether it's worth doing on
!> your subtree.
!>
!> Build (see ../README.md for the full recipe):
!>   gfortran -c bmm_capi.f90 -I<path-to-bmm-repo> -I<path-to-bmm-repo>/osnf \
!>       -I<path-to-bmm-repo>/sce -I<path-to-bmm-repo>/opt -O3 -fPIC -o bmm_capi.o
!>   gfortran -shared -fPIC -o libbmm.so bmm_capi.o \
!>       <path-to-bmm-repo>/bin_microphysics_module.o \
!>       <path-to-bmm-repo>/b_micro_lib.a <path-to-bmm-repo>/opt/optics.a \
!>       <path-to-bmm-repo>/osnf/osnf_lib.a <path-to-bmm-repo>/sce/sce_micro_lib.a \
!>       <path-to-bmm-repo>/sce/sce_module.o -lnetcdff
module bmm_capi
    use iso_c_binding
    implicit none
contains

    !> Run one BMM case from a namelist file on disk, writing output to the
    !> netCDF path specified inside that namelist (outputfile=...).
    !> Returns 0 on success, nonzero on error (mirrors error_stop conditions
    !> inside bin_microphysics_module.f90 -- those currently call Fortran
    !> `error stop`, which will abort the whole process rather than return
    !> here; that's another reason process-per-call is the robust choice).
    !>
    !> nmlfile_c: null-terminated C string, path to a BMM namelist file.
    function bmm_run_c(nmlfile_c) result(ierr) bind(C, name="bmm_run_c")
        use bmm, only : read_in_bmm_namelist, initialise_bmm_arrays, bmm_driver, io1, &
                        sce_flag, hm_flag, break_flag, mode1_flag, mode2_flag, &
                        psurf, tsurf, q_read, theta_read, rh_read, z_read, &
                        time_chamber, press_chamber, temp_chamber, qtot_chamber, &
                        runtime, dt, zinit, tpert, use_prof_for_tprh, &
                        winit, winit2, amplitude2, tinit, pinit, &
                        rhinit, radinit, bubble_flag, &
                        microphysics_flag, ice_flag, bin_scheme_flag, vent_flag, &
                        kappa_flag, updraft_type, adiabatic_prof, &
                        vert_ent, z_ctop, ent_rate, &
                        n_levels_s, n_levels_c, alpha_therm, alpha_cond, alpha_therm_ice, &
                        alpha_dep, n_intern, n_mode, n_sv, sv_flag, n_bins, n_comps, &
                        n_aer1, d_aer1, sig_aer1, dmina, dmaxa, mass_frac_aer1, molw_core1, &
                        density_core1, nu_core1, kappa_core1, org_content1, molw_org1, &
                        kappa_org1, density_org1, delta_h_vap1, nu_org1, log_c_star1, &
                        BIN_MOVING_CENTRE, BIN_CHEN_LAMB
        implicit none
        character(kind=c_char), dimension(*), intent(in) :: nmlfile_c
        integer(c_int) :: ierr
        character(len=200) :: nmlfile
        logical :: fixed_grid_required
        integer :: i

        ! C string -> Fortran string
        nmlfile = ' '
        do i = 1, 200
            if (nmlfile_c(i) == c_null_char) exit
            nmlfile(i:i) = nmlfile_c(i)
        end do

        ierr = 0
        call read_in_bmm_namelist(trim(nmlfile))

        fixed_grid_required = (sce_flag .gt. 0) .or. &
            (bin_scheme_flag .eq. BIN_MOVING_CENTRE) .or. &
            (bin_scheme_flag .eq. BIN_CHEN_LAMB)
        if (fixed_grid_required) then
            ! Not supported through this minimal C API (SCE fixed-grid setup
            ! needs the extra read_in_sce_namelist / write_sce_grid_to_bmm /
            ! project_initial_bmm_to_fixed_grid calls from main.f90). Use
            ! bin_scheme_flag=0 (BIN_FULL_MOVING) for surrogate training data,
            ! which is also the cheapest/fastest scheme.
            ierr = 1
            return
        endif

        call initialise_bmm_arrays(psurf, tsurf, q_read, theta_read, rh_read, z_read, &
                    time_chamber, press_chamber, temp_chamber, qtot_chamber, &
                    runtime, dt, zinit, tpert, use_prof_for_tprh, &
                    winit, tinit, pinit, &
                    rhinit, radinit, bubble_flag, &
                    microphysics_flag, ice_flag, bin_scheme_flag, vent_flag, &
                    kappa_flag, updraft_type, adiabatic_prof, vert_ent, z_ctop, &
                    ent_rate, n_levels_s, n_levels_c, &
                    alpha_therm, alpha_cond, alpha_therm_ice, &
                    alpha_dep, n_intern, n_mode, n_sv, sv_flag, n_bins, n_comps, &
                    n_aer1, d_aer1, sig_aer1, dmina, dmaxa, mass_frac_aer1, molw_core1, &
                    density_core1, nu_core1, kappa_core1, org_content1, molw_org1, &
                    kappa_org1, density_org1, delta_h_vap1, nu_org1, log_c_star1, sce_flag)

        io1%new_file = .true.
        call bmm_driver(sce_flag, hm_flag, break_flag, mode1_flag, mode2_flag)
    end function bmm_run_c

end module bmm_capi
