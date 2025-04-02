using MLTruncate.Hamiltonian

using Base.Iterators, LinearAlgebra

trivial_sub_matrices(gen_matrix, spaces) = map(gen_matrix, spaces)

trivial_sub_hamiltonians(space, subspaces, is_sparse::Bool=true) = trivial_sub_matrices(subspaces) do subspace
    hamiltonian(space, subspace, is_sparse)
end

size = 0.1
coupling = 1
k = 0
energies = (3.1, 3.5, 3.9)

space = Phi4Impl(size, coupling)

for k in (-1, 0, 2)
    for symmetrisation in Set(MaybeParity)
        sym_label = isnothing(symmetrisation) ? "X-all" : "X-$symmetrisation"
        param_label = "K=$k, $sym_label"

        subspaces, subhamiltonians = zip(sub_hamiltonians(space, energies...; x_symmetrisation = symmetrisation, momentum = k)...)
        all_states = subspaces[end]
        H = subhamiltonians[end]
        max_e = energies[end]

        @testset "State Properties $param_label" begin
            for state in all_states                
                if isnothing(symmetrisation)
                    @test momentum(state) == k
                else
                    @test momentum(state.base_state) == k
                end

                @test free_energy(space, state) <= max_e
            end
        end

        @testset "Hamiltonian Hermitian $param_label" begin
            @test all(isapprox.(H, H'))
        end

        @testset "Energy Ordering $param_label" begin
            expected_subhamiltonians = trivial_sub_hamiltonians(space, subspaces)

            @test subhamiltonians == expected_subhamiltonians

            for (E_min, prev_states, states) in zip(energies, subspaces, drop(subspaces, 1))
                for state in states
                    @test (free_energy(space, state) > E_min) ⊻ (state in prev_states)
                end
            end
        end

        @testset "Degeneracy Test $param_label" begin
            equality_matrix = all_states .== permutedims(all_states)

            @test equality_matrix == I
        end
    end
end