export sub_spaces, sub_hamiltonians, hamiltonian, any_sub_hamiltonians

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
    (; states, coupling, max_energy, hamiltonian = (@. H0 + coupling * V))
end

function any_sub_hamiltonians(space, substates, energies, couplings; is_sparse::Bool=true, ordering_score = nothing)
    if substates isa AbstractArray{<:FieldState}
        substates = Ref(substates)
    end
    
    if !isnothing(ordering_score)
        substates = map(substates) do states
            sort_by_score(ordering_score, states)
        end
    end

    H0 = sub_matrices(FreeHamiltonian(space), substates, is_sparse)
    V = sub_matrices(Phi4Interaction(space), substates, is_sparse)

    Broadcast.broadcasted(assemble_subspacehamiltonian, substates, H0, V, couplings, energies)
end

function sub_hamiltonians(space, eigenspace, energies, couplings; kwargs...)
    """Generate hamiltonians for Phi4 theories with the given couplings, cutoff at the given max energies. Broadcasts over energies, couplings."""

    @info "Generating states"

    substates = sub_spaces(space, eigenspace, energies)

    any_sub_hamiltonians(space, substates, energies, couplings; kwargs...)
end