using MLTruncate

function evaluate_mphys(free_space, max_energy, couplings, K = Int8, N = UInt8)
    plot_name = sanitise("find_critical/Mphys_g=$(couplings)_L=$(size(free_space))_E=$max_energy")
    
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

    plt = plot(couplings, mphys, xlabel = L"g", ylabel = L"m_{\mathrm{phys}}", legend=false, title=L"Critical Point in Physical Mass of $\phi^4$ Theory")
    savefig(plt, "$PLOT_OUT/$plot_name.pdf")
end

evaluate_mphys(FockSpace{Float32}(8), 23, 0:.01:5)