export LadderOperators, min_ladders, with_number_ops, order 

using Combinatorics
import Base: +, zero

struct MonoLadder{N <: Unsigned}
    raising::N
    lowering::N
end

MonoLadder(n) = MonoLadder(n, n)

zero(::Type{MonoLadder{N}}) where N = MonoLadder(zero(N))

function diff_ladder(n_in::N, n_out::N) where {N <: Unsigned}
    MonoLadder(sign_split(n_out, n_in)...)
end

+(operators::MonoLadder...) = MonoLadder(sum(map(op -> op.raising, operators)), sum(map(op -> op.lowering, operators)))

order(ladder::MonoLadder) = ladder.raising + ladder.lowering

occupation_change(ladder::MonoLadder) = signed(ladder.raising) - signed(ladder.lowering)

function scale_factor(space::BoundedFockSpace{E}, momentum::Signed, ladder::MonoLadder)::E where E
    sqrt(free_energy(space, momentum) ^ order(ladder))
end

function field_matrix_element(::BoundedFockSpace{E}, n_in::N, n_out::N, (;raising, lowering)::MonoLadder{N})::E where {E, N <: Unsigned}
    if n_in < lowering || n_out < raising; return 0 end

    n_intermediate = n_in - lowering

    if n_intermediate != n_out - raising; return 0 end
    
    n_low, n_high = n_out > n_in ? (n_in, n_out) : (n_out, n_in)

    symmetry_factor = factorial(raising) * factorial(lowering)

    sqrt(factorial(n_high, n_low)) * factorial(n_low, n_intermediate) / symmetry_factor
end

const LadderOperators{K <: Signed, N <: Unsigned} = Dict{K, MonoLadder{N}}
LadderOperators(pairs) = Dict(pairs...)
number_operators(operator_powers::Pair...) = LadderOperators(k => MonoLadder(n) for (k, n) in operator_powers)

+(operators::LadderOperators...) = mergewith(+, operators...)

function min_ladders(in_state::FockState{K, N}, out_state::FockState{K, N}) where {K, N}
    LadderOperators{K, N}(k => diff_ladder(ns...) for (k, ns) in iter_all(in_state, out_state))
end

function order(ladder::LadderOperators{K, N}) where {K, N}
    sum(order(mono) for (_, mono) in ladder; init=zero(N))
end

function momentum(ladder::LadderOperators{K, N}) where {K, N}
    sum(k * occupation_change(mono) for (k, mono) in pairs(ladder); init=zero(K))
end

function field_matrix_element(space::BoundedFockSpace{E}, in_state::FockState, out_state::FockState, ladders::LadderOperators)::E where E
    product = coupling(space)

    for (k, ladder) in ladders
        factor = field_matrix_element(space, occupation_number(in_state, k), occupation_number(out_state, k), ladder)

        if iszero(factor); return 0 end

        product *= factor / scale_factor(space, k, ladder)
    end

    product
end