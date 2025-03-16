export BoundedFockSpace, FockSpaceImpl, k_unit, free_energy, hamiltonian, dense_hamiltonian, sparse_hamiltonian, hamiltonian_element, diagonal_hamiltonian_element


using Base.Iterators
using .IntPartitions

abstract type BoundedFockSpace{E <: AbstractFloat} end

struct FockSpaceImpl{E <: AbstractFloat} <: BoundedFockSpace{E}
    k_unit::E
end

k_unit(space::FockSpaceImpl) = space.k_unit

function free_energy(space::BoundedFockSpace{E}, momentum::Integer)::E where {E <: AbstractFloat}
    sqrt(1 + (k_unit(space) * momentum)^2)
end

function free_energy(space::BoundedFockSpace{E}, state::FockStateDiff)::E where {E <: AbstractFloat}
    sum((n * free_energy(space, k) for (k, n) in state); init = zero(E))
end

function free_energy(space::BoundedFockSpace{E}, state::AnySymFockState)::E where {E <: AbstractFloat}
    free_energy(space, state.state)
end

function free_energy(space::BoundedFockSpace{E}, momenta)::E where {E <: AbstractFloat}
    sum((free_energy(space, k) for k in momenta); init = zero(E))
end

function diagonal_hamiltonian_element(space::BoundedFockSpace{E}, state::FockState)::E where {E <: AbstractFloat}
    free_energy(space, state)
end

diagonal_hamiltonian_element(space::BoundedFockSpace) = state -> diagonal_hamiltonian_element(space, state)
 
function hamiltonian_element(space::BoundedFockSpace{E}, in_state::FockState, out_state::FockState)::E where {E <: AbstractFloat}
    if in_state == out_state
        diagonal_hamiltonian_element(space, in_state)
    else
        0
    end
end

hamiltonian_element(space::BoundedFockSpace) = (i, o) -> hamiltonian_element(space, i, o) 

function hamiltonian_element(space::BoundedFockSpace{E}, in_state::AnyFockState, out_state::AnyFockState)::E where {E <: AbstractFloat}
    matrix_element(hamiltonian_element(space), in_state, out_state)
end

function diagonal_hamiltonian_element(space::BoundedFockSpace{E}, state::AnyFockState)::E where {E <: AbstractFloat}
    diagonal_matrix_element(diagonal_hamiltonian_element(space), hamiltonian_element(space), state)
end

function dense_hamiltonian(space::BoundedFockSpace{E}, states::Vector{F}) where {E <: AbstractFloat, F <: AnyFockState}
    hamiltonian_element.([space], states, permutedims(states))
end

function sparse_hamiltonian(space::BoundedFockSpace{E}, states::Vector{F}) where {E <: AbstractFloat, F <: AnyFockState}
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

function hamiltonian(space::BoundedFockSpace{E}, states::Vector{F}, is_sparse::Bool = true) where {E <: AbstractFloat, F <: AnyFockState}    
    if is_sparse
        sparse_hamiltonian(space, states)
    else
        dense_hamiltonian(space, states)
    end
end