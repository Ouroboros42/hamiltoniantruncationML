export FockState, FockStateDiff, SymmetrisedFockState, AntiSymmetrisedFockState, AnyFockState, momentum, occupation_number, symmetrise_state, symmetrise_states

import Base: ==

using MLStyle

const FockStateDiff{K, N} = Dict{K, N} where {K <: Signed, N <: Integer}
const FockState{K, N} = FockStateDiff{K, N} where {K <: Signed, N <: Unsigned}

function FockState(occupation_numbers::Pair{K, N}...) where {K <: Signed, N <: Signed}
    Dict(k => force_unsigned(n) for (k, n) in occupation_numbers)
end

occupation_number(state::FockState{K, N}, momentum::K) where {K <: Signed, N <: Unsigned} = get(state, momentum, zero(N))

momentum(state::FockState{K, N}) where {K <: Signed, N <: Unsigned} = sum(K(n) * k for (n, k) in state; init=zero(K))

iter_all(states::FockState...) = ((k, map(state -> occupation_number(state, k), states)) for k in union(map(keys, states)...))

function add_particle!(state::FockState{K, N}, momentum::K) where {K <: Signed, N <: Unsigned}
    state[momentum] = occupation_number(state, momentum) + 1
end

function x_flip(state::FockState{K, N}) where { K <: Signed, N <: Unsigned }
    FockState{K, N}(-k => n for (k, n) in state)
end


abstract type AnySymFockState{K <: Signed, N <: Unsigned} end

==(state1::S, state2::S) where { S <: AnySymFockState } = state1.state == state2.state

struct SymmetrisedFockState{K <: Signed, N <: Integer} <: AnySymFockState{K, N}
    state::FockState{K, N}
    flipped_state::FockState{K, N}
    is_x_symmetric::Bool

    function SymmetrisedFockState(state::FockState{K, N}) where { K <: Signed, N <: Unsigned }
        flipped = x_flip(state)
        new{K, N}(state, flipped, state == flipped)
    end
end

function matrix_element(fockstate_matrix, in_state::SymmetrisedFockState, out_state::SymmetrisedFockState)
    # Assumes fockstate_matrix commutes with x-parity

    @match in_state.is_x_symmetric + out_state.is_x_symmetric begin
        0 => fockstate_matrix(in_state.state, out_state.state) + fockstate_matrix(in_state.flipped_state, out_state.state)
        1 => fockstate_matrix(in_state.state, out_state.state) / sqrt(2)
        2 => fockstate_matrix(in_state.state, out_state.state)
    end
end

function diagonal_matrix_element(fockstate_matrix_diagonal, fockstate_matrix, state::SymmetrisedFockState)
    # Assumes fockstate_matrix commutes with x-parity
    
    if state.is_x_symmetric
        fockstate_matrix_diagonal(state.state)
    else
        fockstate_matrix_diagonal(state.state) + fockstate_matrix(state.state, state.flipped_state)
    end
end

struct AntiSymmetrisedFockState{K <: Signed, N <: Integer} <: AnySymFockState{K, N}
    state::FockState{K, N}
    flipped_state::FockState{K, N}
    
    function AntiSymmetrisedFockState(state::FockState{K, N}) where { K <: Signed, N <: Unsigned }
        flipped = x_flip(state)

        if state == flipped
            nothing
        else
            new{K, N}(state, flipped)
        end
    end
end

function matrix_element(fockstate_matrix, in_state::AntiSymmetrisedFockState, out_state::AntiSymmetrisedFockState)
    # Assumes fockstate_matrix commutes with x-parity
    fockstate_matrix(in_state.state, out_state.state) - fockstate_matrix(in_state.flipped_state, out_state.state)
end

function diagonal_matrix_element(fockstate_matrix_diagonal, fockstate_matrix, state::AntiSymmetrisedFockState)
    # Assumes fockstate_matrix commutes with x-parity

    fockstate_matrix_diagonal(state.state) - fockstate_matrix(state.state, state.flipped_state)
end

const AnyFockState{K, N} = Union{
    FockState{K, N},
    AnySymFockState{K, N}
} where {K <: Signed, N <: Unsigned}

StateType(::Nothing, K, N) = FockState{K, N}
StateType(x_symmetrisation::Parity, K, N) = @match x_parity begin
    &Odd => SymmetrisedFockState{K, N}
    &Even => AntiSymmetrisedFockState{K, N}
end

symmetrise_state(state::FockState, ::Nothing) = state
symmetrise_state(state::FockState, x_parity::Parity) = @match x_parity begin
    &Odd => AntiSymmetrisedFockState(state)
    &Even => SymmetrisedFockState(state)
end

function symmetrise_states(states, x_parity::MaybeParity)
    Iterators.filter(!isnothing, Iterators.map(state -> symmetrise_state(state, x_parity), states))
end