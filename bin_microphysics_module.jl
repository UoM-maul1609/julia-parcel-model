module bmm
    using SpecialFunctions
    using Roots
    
    export bmm_driver, initialise_bmm_arrays
    
    ttr=273.15
    
    function bmm_driver(data)
        println("Running the driver...")
        
        # number of time-steps
        nt::Int16 = ceil(data["run_vars"]["runtime"] / bmm.dt)
        for i=1:nt
            #println("Time-step $i of $nt time in seconds is $(i * bmm.dt)")
            bin_microphysics()
        end
        println("done")
    end
    
    function initialise_bmm_arrays(data)
    
        #= !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            set variables and allocate arrays for parcel
           !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        =#
        global z            =data["run_vars"]["zinit"]
        global p            =data["run_vars"]["pinit"]
        global t            =data["run_vars"]["tinit"]
        global w            =data["run_vars"]["winit"]
        global rh           =data["run_vars"]["rhinit"]
        global dt           =data["run_vars"]["dt"]
        global n_bins       = data["aerosol_setup"]["n_bins"]
        global n_modes      = data["aerosol_setup"]["n_mode"]
        global n_comps      = data["aerosol_setup"]["n_comps"]
        global n_intern     = data["aerosol_setup"]["n_intern"]
        global ice_flag     = data["run_vars"]["ice_flag"]
        global n_bin_modew  = n_bins * n_modes
        global n_bin_mode1  = (n_bins+1) * n_modes
        global n_bin_mode   = n_bins * n_modes*(1+ice_flag)
        global imoms        = ice_flag*5
        global dmina        = data["aerosol_spec"]["dmina"]
        global dmaxa        = data["aerosol_spec"]["dmaxa"]
        global density_core1= data["aerosol_spec"]["density_core1"]
        global nu_core1     = data["aerosol_spec"]["nu_core1"]
        global molw_core1   = data["aerosol_spec"]["molw_core1"]
        global kappa_core1  = data["aerosol_spec"]["kappa_core1"]
        
        global d            = zeros(Float64,n_bin_mode1)
        global mbinedges    = zeros(Float64,n_bins+1,n_modes)
        global maer         = zeros(Float64,n_bin_modew)
        global npart        = zeros(Float64,n_bin_modew)
        global npartall     = zeros(Float64,n_bin_mode)
        global mbin         = zeros(Float64,n_bin_modew,n_comps+1)
        global mbinall      = zeros(Float64,n_bin_mode,n_comps+1)
        global rho_core     = zeros(Float64,n_modes)

        global momtemp      = zeros(Float64,n_bin_mode)
        global moments      = zeros(Float64,n_bin_mode,n_comps+imoms)
        global momenttype   = zeros(Float64,n_comps+imoms)

        global rhobin   = zeros(Float64,n_bin_modew,n_comps)
        global nubin    = zeros(Float64,n_bin_modew,n_comps)
        global molwbin  = zeros(Float64,n_bin_modew,n_comps)
        global kappabin = zeros(Float64,n_bin_modew,n_comps)

        global rh_eq    = zeros(Float64,n_bin_modew)
        global rhoat    = zeros(Float64,n_bin_modew)
        global dw       = zeros(Float64,n_bin_modew)
        global da_dt    = zeros(Float64,n_bin_modew)
        global ndrop    = zeros(Float64,n_bin_modew)

        global ecoal    = zeros(Float64,n_bin_mode,n_bin_mode)
        global ecoll    = zeros(Float64,n_bin_mode,n_bin_mode)
        global indexc   = zeros(Float64,n_bin_mode,n_bin_mode)
        global vel      = zeros(Float64,n_bin_mode,n_bin_mode)
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!




        #= !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            calculate the density of aerosol particles within a mode
           !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        =#
        r=size(data["aerosol_spec"]["mass_frac_aer1"])[1]
        c=size(data["aerosol_spec"]["mass_frac_aer1"][1])[1]
        global mass_frac_aer1=reshape(
            reduce(vcat,data["aerosol_spec"]["mass_frac_aer1"]),(c,r))
        for i=1:n_modes
            var1=sum(mass_frac_aer1[i,:] ./ 
                data["aerosol_spec"]["density_core1"])
            rho_core[i] = 1.0 ./ var1
        end
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


        #= !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            set-up size distribution
           !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        =#
        r=size(data["aerosol_spec"]["n_aer1"])[1]
        c=size(data["aerosol_spec"]["n_aer1"][1])[1]
        global n_aer1=reshape(
            reduce(vcat,data["aerosol_spec"]["n_aer1"]),(c,r))
        global d_aer1=reshape(
            reduce(vcat,data["aerosol_spec"]["d_aer1"]),(c,r))
        global sig_aer1=reshape(
            reduce(vcat,data["aerosol_spec"]["sig_aer1"]),(c,r))
        for k=1:n_modes
            global idum=k
            num1=lognormal_n_between_limits(n_aer1[:,k],d_aer1[:,k],sig_aer1[:,k],
                n_intern,dmina,dmaxa)
            number_per_bin = num1 / n_bins
            npart[1+(k-1)*n_bins:k*n_bins] .= number_per_bin
            d[1+(k-1)*(n_bins+1)]=dmina # min diameter for this mode
            
            for i=1:n_bins
                global d_dummy = d[i+(k-1)*(n_bins+1)]
                global n_dummy = number_per_bin*(1.0-1.e-5)
                d[i+1+(k-1)*(n_bins+1)]=find_zero(find_upper_diameter, 
                    d[i+(k-1)*(n_bins+1)] )
            end
            
            d[k*(n_bins+1)]=dmaxa # max diameter for this mode
        end
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


        #= !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            aerosol mass - total
           !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        =#
        for k=1:n_modes
            for j=1:n_bins
                i=j+(k-1)*(n_bins+1)
                maer[j+(k-1)*(n_bins)] = pi/6.0*(0.5*(d[i+1]+d[i]))^3*rho_core[k]
            end
        end
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


        #= !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            calculate the mass of each component in a bin, including water
           !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        =#
        for i=1:n_bins
            for j=1:n_modes
                for k=1:n_comps
                    mbin[i+(j-1)*n_bins,k]=maer[i+(j-1)*n_bins]*mass_frac_aer1[j,k]
                    # density in each bin
                    rhobin[i+(j-1)*n_bins,k]=density_core1[k]
                    # nu in each bin
                    nubin[i+(j-1)*n_bins,k]=nu_core1[k]
                    # molw in each bin
                    molwbin[i+(j-1)*n_bins,k]=molw_core1[k]                    
                    # kappa in each bin
                    kappabin[i+(j-1)*n_bins,k]=kappa_core1[k]
                end
            end
        end
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


    end
    
    
    
    
    
    
    #= !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        integrate lognormal
        finds the number of aerosol particles between two limits
        n_aer1,d_aer1,sig_aer1 - the aerosol parameters
        n_intern - the number of modes - of the same composition
        dmin, dmax - the limits
        returns the number between limits
       !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    =#
    function lognormal_n_between_limits(n_aer1,d_aer1,sig_aer1,n_intern,dmin,dmax)
        num1=0.0
        for i=1:n_intern
            num1=num1+n_aer1[i]*(0.5*erfc(-log(dmax/d_aer1[i])/sqrt(2.0)/sig_aer1[i] ) -
                0.5*erfc(-log(dmin/d_aer1[i])/sqrt(2.0)/sig_aer1[i] ) )
        end
        return num1
    end
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    
    #= !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
       calculate the upper diameter of the bin-edge
       in: x - dmax guess 
       return num_calc-num = 0 when root round
       !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    =#
    function find_upper_diameter(x)
        num1=lognormal_n_between_limits(n_aer1[:,idum],d_aer1[:,idum],sig_aer1[:,idum],
            n_intern,d_dummy,x)

        return num1-n_dummy
    end
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    #= !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
       saturation vapour pressure over liquid
       in: t - temperature 
       return saturation vapour pressure over liquid water
       !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    =#
    function svp_liq(t)

        return 611.21*exp((18.678-(t-ttr)/234.5)*(t-ttr)/(257.14+(t-ttr)))
    end
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    #= !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
       saturation vapour pressure over ice
       in: t - temperature 
       return saturation vapour pressure over ice
       !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    =#
    function svp_ice(t)

        return 611.15*exp((23.036-(t-ttr)/333.7)*(t-ttr)/(279.82+(t-ttr)))
    end
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    
    function bin_microphysics()
        #println(myvariable)
        #println(dt)
    end
    
end
