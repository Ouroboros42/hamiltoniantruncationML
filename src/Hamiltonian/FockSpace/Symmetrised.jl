export SymmetrisedFockState, SymmetricState, AntisymmetricState

import Base: iszero

struct SymmetrisedFockState{Sym, F} <: FieldState
    base_state::F
    flipped_state::F
    is_x_symmetric::Bool

    function SymmetrisedFockState{Sym}(state::F) where {F, Sym}
        flipped = x_flipped(state)
        new{Sym, F}(state, flipped, state == flipped)
    end
end

const SymmetricState{F} = SymmetrisedFockState{Even, F}
const AntisymmetricState{F} = SymmetrisedFockState{Odd, F}

representative_fockstate(state::SymmetrisedFockState) = state.base_state

==(state1::S, state2::S) where { S <: SymmetrisedFockState } = state1.base_state == state2.base_state || state1.base_state == state2.flipped_state 

iszero(::FieldState) = false
iszero(state::AntisymmetricState) = state.is_x_symmetric

function element(matrix::NiceMatrix, in_state::SymmetricState, out_state::SymmetricState)
    @match in_state.is_x_symmetric + out_state.is_x_symmetric begin
        &0 => element(matrix, in_state.base_state, out_state.base_state) + element(matrix, in_state.flipped_state, out_state.base_state)
        &1 => element(matrix, in_state.base_state, out_state.base_state) / sqrt(2)
        &2 => element(matrix, in_state.base_state, out_state.base_state)
    end
end

function diagonal_element(matrix::NiceMatrix, state::SymmetricState)
    if state.is_x_symmetric
        diagonal_element(matrix, state.base_state)
    else
        diagonal_element(matrix, state.base_state) + element(matrix, state.base_state, state.flipped_state)
    end
end

function element(matrix::NiceMatrix, in_state::AntisymmetricState, out_state::AntisymmetricState)
    element(matrix, in_state.base_state, out_state.base_state) - element(matrix, in_state.flipped_state, out_state.base_state)
end

function diagonal_element(matrix::NiceMatrix, state::AntisymmetricState)
    diagonal_element(matrix, state.base_state) - element(matrix, state.base_state, state.flipped_state)
end

symmetrise_state(state::FockState, x_parity::Nothing) = state
symmetrise_state(state::FockState, x_parity::Parity) = SymmetrisedFockState{x_parity}(state)

symmetrise_states(states, x_parity::Nothing, filter_zero::Bool = true) = states
function symmetrise_states(states, x_parity::Parity, filter_zero::Bool = true)
    sym_states = (symmetrise_state(state, x_parity) for state in states)

    if filter_zero
        Iterators.filter(!iszero, sym_states)
    else
        sym_states
    end
end