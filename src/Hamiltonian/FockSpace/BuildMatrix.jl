export hamiltonian, dense_hamiltonian, sparse_hamiltonian, hamiltonian_element, diagonal_hamiltonian_element

function diagonal_hamiltonian_element(space::BoundedFockSpace{E}, state::FockState)::E where E
    free_energy(space, state)
end

diagonal_hamiltonian_element(space::BoundedFockSpace) = state -> diagonal_hamiltonian_element(space, state)
 
function hamiltonian_element(space::BoundedFockSpace{E}, in_state::FockState, out_state::FockState)::E where E
    if in_state == out_state
        diagonal_hamiltonian_element(space, in_state)
    else
        0
    end
end

hamiltonian_element(space::BoundedFockSpace) = (in, out) -> hamiltonian_element(space, in, out) 

function hamiltonian_element(space::BoundedFockSpace{E}, in_state::FieldState, out_state::FieldState)::E where E
    matrix_element(hamiltonian_element(space), in_state, out_state)
end

function diagonal_hamiltonian_element(space::BoundedFockSpace{E}, state::FieldState)::E where E
    diagonal_matrix_element(diagonal_hamiltonian_element(space), hamiltonian_element(space), state)
end

function dense_hamiltonian(space::BoundedFockSpace{E}, states::Vector{F}) where {E, F <: FieldState}
    hamiltonian_element.([space], states, permutedims(states))
end

function sparse_hamiltonian(space::BoundedFockSpace{E}, states::Vector{F}) where {E, F <: FieldState}
    n_states = length(states)

    H = SparseMatrixCSC{E, keytype(states)}(undef, n_states, n_states)

    indexed_states = collect(enumerate(states))

    for (i, state) in indexed_states        
        H[i, i] = diagonal_hamiltonian_element(space, state)
    end

    for ((i1, state1), (i2, state2)) in combinations(indexed_states, 2)       
        H[i1, i2] = (H[i2, i1] = hamiltonian_element(space, state1, state2))
    end

    H
end

function hamiltonian(space::BoundedFockSpace{E}, states::Vector{F}, is_sparse::Bool = true) where {E <: AbstractFloat, F <: FieldState}    
    if is_sparse
        sparse_hamiltonian(space, states)
    else
        dense_hamiltonian(space, states)
    end
end