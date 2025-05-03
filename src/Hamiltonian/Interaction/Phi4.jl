export Phi4Space, Phi4Impl, k_unit, coupling, Phi4Interaction

using MLStyle
using Combinatorics
using SparseArrays
import Base.Broadcast: broadcastable

abstract type Phi4Space{E <: AbstractFloat} <: BoundedFockSpace{E} end

struct Phi4Impl{E <: AbstractFloat} <: Phi4Space{E}
    k_unit::E
    coupling::E
end

Phi4Impl(k_unit::E, coupling::Real = zero(E)) where {E <: Real} = Phi4Impl(promoteto(float, k_unit, coupling)...)
Phi4Impl(free_space::BoundedFockSpace{E}, coupling = zero(E)) where E = Phi4Impl(k_unit(free_space), coupling)

k_unit(space::Phi4Impl) = space.k_unit
coupling(space::Phi4Impl) = space.coupling

struct Phi4Interaction{E, F <: BoundedFockSpace{E}} <: NiceMatrix{E}
    space::F
end

has_offdiagonal(::Phi4Interaction) = true

function diagonal_element((; space)::Phi4Interaction{E}, state::FockState{K, N})::E where {E, K <: Signed, N <: Unsigned}
    total = zero(E)

    counts_over_1 = K[]

    for (k, n) in state
        if n < 1; continue end

        push!(counts_over_1, k)

        if n < 2; continue end
        
        total += field_matrix_element(space, state, state, number_operators(K, N, k => 2))
    end

    for (k1, k2) in combinations(counts_over_1, 2)
        total += field_matrix_element(space, state, state, number_operators(K, N, k1 => 1, k2 => 1))
    end

    return total
end

function element(matrix::Phi4Interaction{E}, in_state::FockState, out_state::FockState)::E where E
    ladders = min_ladders(in_state, out_state) 

    if !iszero(momentum(ladders)); return 0 end

    # Cannot be odd by n_parity
    # Cannot be > 4 for phi^4 interaction
    # Cannot be 2 by momentum conservation
    @match rank(ladders) begin
        &0 => diagonal_element(matrix, in_state)
        &4 => field_matrix_element(matrix.space, in_state, out_state, ladders)
        _ => 0
    end
end

hamiltonian(space::Phi4Space) = LinearCombination(FreeHamiltonian(space), (coupling(space), Phi4Interaction(space)))