export LadderOperators, min_ladders, with_number_ops, rank 

using Combinatorics
import Base: +, zero

struct MonoLadder{N <: Unsigned}
    raising::N
    lowering::N
end

MonoLadder(n) = MonoLadder(n, n)

zero(::Type{MonoLadder{N}}) where N = MonoLadder(zero(N))

diff_ladder(n_in, n_out) = MonoLadder(sign_split(n_out, n_in)...)

+(op1::MonoLadder, op2::MonoLadder) = MonoLadder(op1.raising + op2.raising, op1.lowering + op2.lowering)

rank(ladder::MonoLadder) = ladder.raising + ladder.lowering

occupation_change(ladder::MonoLadder) = signed(ladder.raising) - signed(ladder.lowering)

const LadderOperators{K <: Signed, N <: Unsigned} = Dict{K, MonoLadder{N}}
number_operators(::Type{K}, ::Type{N}, operator_powers::Pair...) where {K, N} = LadderOperators{K, N}(convert(K, k) => MonoLadder(convert(N, n)) for (k, n) in operator_powers)

+(operators::LadderOperators...) = mergewith(+, operators...)

function min_ladders(in_state::FockState{K, N}, out_state::FockState{K, N}) where {K, N}
    LadderOperators{K, N}(k => diff_ladder(ns...) for (k, ns) in iter_all(in_state, out_state))
end

function rank(ladder::LadderOperators{K, N}) where {K, N}
    sum(rank(mono) for (_, mono) in ladder; init=zero(N))
end

function momentum(ladder::LadderOperators{K, N}) where {K, N}
    sum(k * occupation_change(mono) for (k, mono) in pairs(ladder); init=zero(K))
end

function field_matrix_element(::BoundedFockSpace{E}, n_in, n_out, (;raising, lowering)::MonoLadder)::E where E
    n_intermediate = n_in - lowering

    if !(0 <= n_intermediate == n_out - raising); return 0 end
    
    n_low, n_high = n_out > n_in ? (n_in, n_out) : (n_out, n_in)

    symmetry_factor = factorial(raising) * factorial(lowering)

    sqrt(factorial(n_high, n_low)) * factorial(n_low, n_intermediate) / symmetry_factor
end

PHI4CONST = factorial(4) / (8 * pi)

function field_matrix_element(space::BoundedFockSpace{E}, in_state::FockState, out_state::FockState, ladders::LadderOperators)::E where E
    product = k_unit(space) * convert(E, PHI4CONST) 
    energy_product = one(E)

    for (k, ladder) in ladders
        factor = field_matrix_element(space, occupation_number(in_state, k), occupation_number(out_state, k), ladder)

        if iszero(factor); return 0 end

        product *= factor

        energy_product *= free_energy(space, k) ^ rank(ladder)
    end

    product / sqrt(energy_product)
end