export sub_spaces, sub_hamiltonians

function sub_spaces(space::BoundedFockSpace, energies...; kwargs...)
    map(energies) do energy
        states = collect(generate_states(space, energy; kwargs...))
        if isempty(states)
            throw("Empty subspace created at energy $energy")
        end
        states
    end
end

indices_of(items, sequence) = findfirst.(.==(items), Ref(sequence))

function sub_matrices(gen_matrix, (substates..., allstates)::NTuple{N, Vector}) where N
    largest_matrix = gen_matrix(allstates)

    smaller_matrices = map(substates) do states

        outer_indices = indices_of(states, allstates)
        if any(isnothing, outer_indices)
            throw("Substates not properly nested")
        end
        largest_matrix[outer_indices, outer_indices]
    end

    (smaller_matrices..., largest_matrix)
end

function sub_hamiltonians(space, energies...; is_sparse::Bool=true, kwargs...)
    substates = sub_spaces(space, energies...; kwargs...)

    hamiltionians = sub_matrices(substates) do states
        hamiltonian(space, states, is_sparse)
    end

    map(=>, substates, hamiltionians) 
end