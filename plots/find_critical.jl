using MLTruncate

function evaluate_mphys(free_space, max_energy, couplings, K = Int8, N = UInt8)
    plot_name = sanitise("find_critical/Mphys_g=$(couplings)_L=$(size(free_space))_E=$(join(max_energy, "-"))")
    
    (; E0, E1) = cache("$PLOT_CACHE/$plot_name.bson") do
        map((E0 = Even, E1 = Odd)) do Pn
            eigenspace = EigenSpace{K, N}(x_symmetrisation=Even, n_parity = Pn)

            subhams = sub_hamiltonians(free_space, eigenspace, max_energy, couplings)

            map(subhams) do (; hamiltonian)
                groundstate(hamiltonian)[2]
            end
        end
    end
    
    mphys = E1 .- E0

    plot_kwargs = (;
        xlabel = L"\tilde g", ylabel = L"\tilde m_{\mathrm{phys}}", legendtitle=L"\tilde E_{\mathrm{max}}",
        label = max_energy
    )

    plt = plot(couplings, mphys; plot_kwargs...)
    
    std_savefig(plt, plot_name)
end

evaluate_mphys(FockSpace{Float32}(8), [11 14 17 20 23], 0:.01:5)