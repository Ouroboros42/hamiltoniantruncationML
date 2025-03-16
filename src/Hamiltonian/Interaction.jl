export Phi4Space, Phi4Impl, k_unit, coupling

using MLStyle
using Combinatorics
using SparseArrays
import Base.Broadcast: broadcastable

abstract type Phi4Space{E <: AbstractFloat} <: BoundedFockSpace{E} end

struct Phi4Impl{E <: AbstractFloat} <: Phi4Space{E}
    k_unit::E
    coupling::E
end

Phi4Impl(k_unit::Real, coupling::Real) = Phi4Impl(promote(k_unit, coupling)...)

k_unit(space::Phi4Impl) = space.k_unit
coupling(space::Phi4Impl) = space.coupling

function diagonal_hamiltonian_element(space:: Phi4Space{E}, state::FockState{K, N})::E where {E <: AbstractFloat, K <: Signed, N <: Unsigned}
    total = free_energy(space, state)

    counts_over_1 = K[]

    for (k, n) in state
        if n < 1; continue end

        push!(counts_over_1, k)

        if n < 2; continue end
        
        total += field_matrix_element(space, state, state, number_operators(k => N(2)))
    end

    for (k1, k2) in combinations(counts_over_1, 2)
        total += field_matrix_element(space, state, state, number_operators(k1 => N(1), k2 => N(1)))
    end

    return total
end

function hamiltonian_element(space::Phi4Space{E}, in_state::FockState, out_state::FockState)::E where {E <: AbstractFloat}
    ladders = min_ladders(in_state, out_state) 

    if momentum(ladders) != 0; return 0 end

    # Cannot be odd by n_parity
    # Cannot be > 4 for phi^4 interaction
    # Cannot be 2 by momentum conservation
    @match order(ladders) begin
        0 => diagonal_hamiltonian_element(space, in_state)
        4 => field_matrix_element(space, in_state, out_state, ladders)
        _ => 0
    end
end
