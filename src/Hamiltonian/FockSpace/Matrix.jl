export compute

abstract type NiceMatrix{E} end

has_offdiagonal(::NiceMatrix) = false

function compute(::Type{Out}, matrix::NiceMatrix{E}, states, args...; kwargs...) where {E, Out <: AbstractMatrix{E}}
    return compute!(Out(undef, length(states), length(states)), matrix, states, args...; kwargs...)
end
compute_dense(matrix::NiceMatrix{E}, args...; kwargs...) where E = compute(Matrix{E}, matrix, args...; kwargs...)
compute_sparse(matrix::NiceMatrix{E}, states, args...; kwargs...) where E = compute(SparseMatrixCSC{E, keytype(states)}, matrix, states, args...; kwargs...)
compute(args...; is_sparse::Bool = true, kwargs...) = is_sparse ? compute_sparse(args...; kwargs...) : compute_dense(args...; kwargs...)

function compute!(output, matrix::NiceMatrix, states)
    indexed_states = collect(enumerate(states))

    for (i, state) in indexed_states        
        output[i, i] = diagonal_element(matrix, state)
    end

    if has_offdiagonal(matrix)
        for ((i_in, in_state), (i_out, out_state)) in combinations(indexed_states, 2)
            set_offdiagonal!(output, matrix, i_in, in_state, i_out, out_state)
        end
    end

    output
end

function compute!(output, matrix::NiceMatrix, states, base_computed, base_states)
    base_indices = keys(base_states)

    output[base_indices, base_indices] = base_computed
    
    extend_indices = map(i -> i + length(base_states), keys(states))

    compute!(matrix, states, output[extend_indices, extend_indices])

    if has_offdiagonal(matrix)
        for (i_base, base_state) in zip(base_indices, base_states)
            for (i_extend, extend_state) in zip(extend_indices, extend_states)
                set_offdiagonal!(output, matrix, i_base, base_state, i_extend, extend_state)
            end
        end
    end

    output
end

function set_offdiagonal!(output, matrix::NiceMatrix, i_in, in_state, i_out, out_state)
    elem = element(matrix, in_state, out_state)

    output[i_out, i_in] = elem
    output[i_in, i_out] = conj(elem)
end

# Default implementation (zero matrix). Allows only diagonal or off-diagonal elements to be defined.
element(matrix::NiceMatrix{E}, in_state::FockState, out_state::FockState) where E = in_state == out_state ? diagonal_element(matrix, in_state) : zero(E)
diagonal_element(matrix::NiceMatrix{E}, state::FockState) where E = zero(E)

# Simple currying to convert to functions of only states
element(matrix::NiceMatrix) = (in_state, out_state) -> element(matrix, in_state, out_state)
diagonal_element(matrix::NiceMatrix) = (state) -> diagonal_element(matrix, state)

const VarTuple{T} = NTuple{N, T} where N

struct LinearCombination{E, T <: VarTuple{Tuple{E, NiceMatrix{E}}}} <: NiceMatrix{E}
    terms::T
end

as_scaled_matrix(matrix::NiceMatrix) = as_scaled_matrix((1, matrix))
as_scaled_matrix((scale, matrix)::Tuple{Number, NiceMatrix{E}}) where E = (convert(E, scale), matrix)

function LinearCombination(terms::Union{NiceMatrix{E}, Tuple{Any, NiceMatrix{E}}}...) where E
    scaled_terms = map(as_scaled_matrix, terms)
    LinearCombination{E, typeof(scaled_terms)}(scaled_terms) 
end

has_offdiagonal(comb::LinearCombination) = any(has_offdiagonal(matrix) && !iszero(coeff) for (coeff, matrix) in comb.terms)

apply_linear(func, comb::LinearCombination{E}) where E = sum(coeff * func(matrix) for (coeff, matrix) in comb.terms; init=zero(E))

element(comb::LinearCombination, in_state::FockState, out_state::FockState) = apply_linear(comb) do term; element(term, in_state, out_state) end
diagonal_element(comb::LinearCombination, state::FockState) = apply_linear(comb) do term; diagonal_element(term, state) end


