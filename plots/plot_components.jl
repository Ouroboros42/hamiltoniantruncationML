using MLTruncate

function plot_components(space, subspace, max_energies, couplings)
    plot_name = "components/N-$(subspace.n_parity)_E=$(max_energies)_g=$(couplings)"

    components = std_cache(plot_name) do 
        subhams = sub_hamiltonians(space, subspace, max_energies, couplings; ordering_score = -free_energy(space))

        hcat(map(subhams) do (; coupling, hamiltonian)
            normalise_components(groundstate(hamiltonian))[2:end]
        end...)
    end
    
    label = hcat(map(couplings) do coupling
        latexstring("g = $coupling")
    end...)

    plt = groupedbar(components; xlabel = "States ordered by free energy", ylabel = "Magnitude of component", label)

    std_savefig(plt, plot_name)
end

plot_components(FockSpace{Float32}(8), EigenSpace{Int8, UInt8}(0, Even, Even), 20, 2)