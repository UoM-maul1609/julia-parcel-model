module bmm
    export bmm_driver, initialise_bmm_arrays
    global myvariable;
    
    
    
    function bmm_driver(data)
        println("Running the driver...")
        
        # number of time-steps
        nt::Int16 = ceil(data["run_vars"]["runtime"] / data["run_vars"]["dt"])
        dt = data["run_vars"]["dt"]
        for i=1:nt
            println("Time-step $i of $nt time in seconds is $(i * dt)")
            bin_microphysics()
        end
        println("done")
    end
    
    function initialise_bmm_arrays(data)
        global myvariable = "Hello there!"
    end
    
    
    function bin_microphysics()
        println(myvariable)
    
    end
    
end
