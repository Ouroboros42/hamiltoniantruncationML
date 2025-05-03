using MLTruncate.Hamiltonian
using Plots

function evaluate_mphys(free_space, max_energy, couplings, K = Int8, N = UInt8)
    (E0, E1) = map((Even, Odd)) do Pn
        eigenspace = EigenSpace{K, N}(x_symmetrisation=Even, n_parity = Pn)

        states = collect(generate_states(free_space, eigenspace, max_energy))
    
        @info "Generated $(length(states)) N-$Pn states"

        H0 = compute(FreeHamiltonian(free_space), states)

        @info "Computed Free Hamiltonian"

        V = compute(Phi4Interaction(free_space), states)

        @info "Computed Interaction Hamiltonian"

        map(couplings) do coupling
            H = @. H0 + coupling * V

            _, (E0, E1) = spectrum(H, 2)

            E1 - E0
        end
    end

    mphys = E1 .- E0
    
    plot(couplings, [E0 E1 mphys], label = ["E0" "E1" "Mphys"], xlabel = "g", ylabel = "E / m")
end

evaluate_mphys(FockSpaceImpl(Float32(2π / 10)), 15, 0:.1:5)