using MLTruncate

using Base.Iterators, LinearAlgebra

trivial_sub_hamiltonians(space, subspaces, couplings) = @. compute(hamiltonian(space, couplings), subspaces)

size = 70
coupling = 1
k = 0
energies = (3.1, 3.5, 3.9)

space = FockSpace(size)

for k in (-1, 0, 2)
    for x_symmetrisation in Set(MaybeParity)
        eigspace = EigenSpace(k; x_symmetrisation)

        subspaces, subhamiltonians = zip(((states, hamiltonian) for (; states, hamiltonian) in sub_hamiltonians(space, eigspace, energies, coupling))...)
        all_states = subspaces[end]
        H = subhamiltonians[end]
        max_e = energies[end]

        @testset "State Properties $eigspace" begin
            for state in all_states                
                if isnothing(x_symmetrisation)
                    @test momentum(state) == k
                else
                    @test momentum(state.base_state) == k
                end

                @test free_energy(space, state) <= max_e
            end
        end

        @testset "Hamiltonian Hermitian $eigspace" begin
            @test all(isapprox.(H, H'))
        end

        @testset "Energy Ordering $eigspace" begin
            expected_subhamiltonians = trivial_sub_hamiltonians(space, subspaces, coupling)

            @test subhamiltonians == expected_subhamiltonians

            for (E_min, prev_states, states) in zip(energies, subspaces, drop(subspaces, 1))
                for state in states
                    @test (free_energy(space, state) > E_min) ⊻ (state in prev_states)
                end
            end
        end

        @testset "Degeneracy Test $eigspace" begin
            equality_matrix = all_states .== permutedims(all_states)

            @test equality_matrix == I
        end
    end
end