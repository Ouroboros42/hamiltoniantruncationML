using MLTruncate

function evaluate_mphys(free_space, max_energy, couplings, K = Int8, N = UInt8)
    plot_name = sanitise("Mphys_g=$(couplings)_L=$(size(free_space))_E=$max_energy")
    
    (; E0, E1) = cache("$PLOT_CACHE/$plot_name.bson") do
        map((E0 = Even, E1 = Odd)) do Pn
            eigenspace = EigenSpace{K, N}(x_symmetrisation=Even, n_parity = Pn)

            subhams = sub_hamiltonians(free_space, eigenspace, max_energy, couplings, include_states=false)

            map(subhams) do (; hamiltonian)
                components, E = groundstate(hamiltonian)
                E
            end
        end
    end
    
    mphys = E1 .- E0

    plt = plot(couplings, mphys, xlabel = "g", ylabel = "Mphys", legend=false)
    savefig(plt, "$PLOT_OUT/$plot_name.png")
end

evaluate_mphys(FockSpace{Float32}(8), 22, 0:.01:5)