export sub_spaces, sub_hamiltonians, hamiltonian

using Logging

function sub_spaces(space::BoundedFockSpace, eigenspace::EigenSpace, energies...)
    map(energies) do energy
        states = collect(generate_states(SubSpace(space, eigenspace, energy)))
        
        if isempty(states)
            throw("Empty subspace created at energy $energy")
        end

        states
    end
end

indices_of(items, sequence) = findfirst.(.==(items), Ref(sequence))

function sub_matrices(gen_matrix, (substates..., allstates)::NTuple{N, Vector}) where N
    @info "Computing matrix for $(length(allstates)) states"

    largest_matrix = gen_matrix(allstates)

    @info "Matrix computed"

    smaller_matrices = map(substates) do states

        outer_indices = indices_of(states, allstates)
        if any(isnothing, outer_indices)
            throw("Substates not properly nested")
        end
        largest_matrix[outer_indices, outer_indices]
    end

    (smaller_matrices..., largest_matrix)
end

function sub_hamiltonians(space, eigenspace, energies...; is_sparse::Bool=true)
    @info "Generating states"

    substates = sub_spaces(space, eigenspace, energies...)

    hamiltionians = sub_matrices(substates) do states
        hamiltonian(space, states, is_sparse)
    end

    map(=>, substates, hamiltionians) 
end

hamiltonian(space, eigenspace, energy; is_sparse::Bool=true) = only(sub_hamiltonians(space, eigenspace, energy; is_sparse)).second