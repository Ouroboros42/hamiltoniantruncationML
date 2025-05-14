using MLTruncate

function plot_components(space, subspace, max_energies, couplings)
    subhams = sub_hamiltonians(space, subspace, max_energies, couplings; ordering_score = -free_energy(space))

    components = hcat(map(subhams) do (; coupling, hamiltonian)
        normalise_components(groundstate(hamiltonian))[2:end]
    end...)

    label = hcat(map(couplings) do coupling
        latexstring("g = $coupling")
    end...)

    groupedbar(components; xlabel = "States ordered by free energy", ylabel = "Magnitude of component", label)
end

plot_components(FockSpace{Float32}(8), EigenSpace{Int8, UInt8}(0, Even, Even), 10, (0.5, 1, 2))