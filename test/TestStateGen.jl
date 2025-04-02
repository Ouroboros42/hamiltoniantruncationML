using MLTruncate.Hamiltonian

using Base.Iterators, LinearAlgebra

@testset "Known Properties" begin
    size = 0.1
    max_e = 4
    k = 0
    coupling = 1

    space = Phi4Impl(size, coupling)
    states = collect(generate_states(space, max_e))

    for state in states
        @test momentum(state) == k

        @test free_energy(space, state) <= max_e
    end
    
    H = sparse_hamiltonian(space, states)

    @test all(isapprox.(H, H'))
end

trivial_sub_matrices(gen_matrix, spaces) = map(gen_matrix, spaces)

trivial_sub_hamiltonians(space, subspaces, is_sparse::Bool=true) = trivial_sub_matrices(subspaces) do subspace
    hamiltonian(space, subspace, is_sparse)
end

@testset "Energy Ordering" begin
    size = 0.1

    space = FockSpaceImpl(size)

    energies = (3.1, 3.5, 3.9)

    for symmetrisation in (Odd, Even)
        subspaces, subhamiltonians = zip(sub_hamiltonians(space, energies...; x_symmetrisation = symmetrisation)...)

        expected_subhamiltonians = trivial_sub_hamiltonians(space, subspaces)

        @test subhamiltonians == expected_subhamiltonians

        for (E_min, prev_states, states) in zip(energies, subspaces, drop(subspaces, 1))
            for state in states
                @test (free_energy(space, state) > E_min) ⊻ (state in prev_states)
            end
        end

        all_states = subspaces[end]

        equality_matrix = all_states .== permutedims(all_states)

        @test equality_matrix == I
    end
end