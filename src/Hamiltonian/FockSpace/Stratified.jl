export sub_spaces, sub_hamiltonians, hamiltonian

function sub_spaces(space::FockSpace, eigenspace::EigenSpace, energies)
    map(energies) do energy
        states = collect(generate_states(space, eigenspace, energy))
        
        if isempty(states)
            throw("Empty subspace created at energy $energy")
        end

        states
    end
end

indices_of(items, sequence) = findfirst.(.==(items), Ref(sequence))

function sub_matrices(matrix::NiceMatrix, substates, is_sparse::Bool = true)
    allstates = union(substates...)

    @info "Computing matrix for $(length(allstates)) states"

    largest_matrix = compute(matrix, allstates; is_sparse)

    @info "Matrix computed"

    map(substates) do states
        outer_indices = indices_of(states, allstates)

        if any(isnothing, outer_indices)
            throw("Substates not properly nested")
        end

        largest_matrix[outer_indices, outer_indices]
    end
end

function assemble_subspacehamiltonian(states, H0, V, coupling, max_energy)
    (; coupling, max_energy, states, hamiltonian = (@. H0 + coupling * V))
end

function sub_hamiltonians(space, eigenspace, energies, couplings; is_sparse::Bool=true)
    """Generate hamiltonians for Phi4 theories with the given couplings, cutoff at the given max energies. Broadcasts over energies, couplings."""

    @info "Generating states"

    substates = sub_spaces(space, eigenspace, energies)

    if substates isa AbstractArray{<:FieldState}
        substates = Ref(substates)
    end

    H0 = sub_matrices(FreeHamiltonian(space), substates, is_sparse)
    V = sub_matrices(Phi4Interaction(space), substates, is_sparse)

    assemble_subspacehamiltonian.(substates, H0, V, couplings, energies)
end