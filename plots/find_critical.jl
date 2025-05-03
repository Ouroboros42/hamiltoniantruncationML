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

            _, E = groundstate(H)

            E
        end
    end

    mphys = E1 .- E0
    
    plot(couplings, mphys, xlabel = "g", ylabel = "Mphys")
end

evaluate_mphys(FockSpaceImpl(Float32(2π / 8)), 20, 0:.01:5)