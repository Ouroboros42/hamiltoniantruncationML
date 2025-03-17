abstract type WrappedFockState{F <: FockState} <: FieldState end

==(state1::S, state2::S) where { S <: WrappedFockState } = state1.base_state == state2.base_state

struct SymmetrisedFockState{F} <: WrappedFockState{F}
    base_state::F
    flipped_state::F
    is_x_symmetric::Bool

    function SymmetrisedFockState(state::F) where {F}
        flipped = x_flipped(state)
        new{F}(state, flipped, state == flipped)
    end
end

struct AntiSymmetrisedFockState{F} <: WrappedFockState{F}
    base_state::F
    flipped_state::F
    
    function AntiSymmetrisedFockState(state::F) where {F}
        flipped = x_flipped(state)

        if state == flipped
            nothing
        else
            new{F}(state, flipped)
        end
    end
end

# Matrix element functions assume fockstate_matrix commutes with x-parity 

function matrix_element(fockstate_matrix, in_state::SymmetrisedFockState, out_state::SymmetrisedFockState)
    @match in_state.is_x_symmetric + out_state.is_x_symmetric begin
        0 => fockstate_matrix(in_state.base_state, out_state.base_state) + fockstate_matrix(in_state.flipped_state, out_state.base_state)
        1 => fockstate_matrix(in_state.base_state, out_state.base_state) / sqrt(2)
        2 => fockstate_matrix(in_state.base_state, out_state.base_state)
    end
end

function diagonal_matrix_element(fockstate_matrix_diagonal, fockstate_matrix, state::SymmetrisedFockState)
    if state.is_x_symmetric
        fockstate_matrix_diagonal(state.base_state)
    else
        fockstate_matrix_diagonal(state.base_state) + fockstate_matrix(state.base_state, state.flipped_state)
    end
end

function matrix_element(fockstate_matrix, in_state::AntiSymmetrisedFockState, out_state::AntiSymmetrisedFockState)
    fockstate_matrix(in_state.base_state, out_state.base_state) - fockstate_matrix(in_state.flipped_state, out_state.base_state)
end

function diagonal_matrix_element(fockstate_matrix_diagonal, fockstate_matrix, state::AntiSymmetrisedFockState)
    fockstate_matrix_diagonal(state.base_state) - fockstate_matrix(state.base_state, state.flipped_state)
end

symmetrise_state(state::FockState, ::Nothing) = state
symmetrise_state(state::FockState, x_parity::Parity) = @match x_parity begin
    &Odd => AntiSymmetrisedFockState(state)
    &Even => SymmetrisedFockState(state)
end

function symmetrise_states(states, x_parity::MaybeParity)
    Iterators.filter(!isnothing, Iterators.map(state -> symmetrise_state(state, x_parity), states))
end