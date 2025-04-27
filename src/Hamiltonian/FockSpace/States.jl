export FockState, n_particles, n_parity, momentum, iter_all, DictFockState, relevant_momenta, occupation_number, x_flipped, x_parity, frozen

import Base: ==, pairs, iterate, length, ImmutableDict, print

using MLStyle

abstract type FieldState end

abstract type FockState{K <: Signed, N <: Unsigned} <: FieldState end

n_particles(state::FockState{K, N}) where {K, N} = sum(n for (k, n) in state; init=zero(N))
n_parity(state::FockState) = number_parity(n_particles(state))

function momentum(state::FockState{K}) where K
    sum(K(n) * k for (k, n) in state; init=zero(K))
end

pairs(state::FockState) = (k => occupation_number(state, k) for k in relevant_momenta(state))
iter_all(states::FockState...) = ((k, map(state -> occupation_number(state, k), states)) for k in union(map(relevant_momenta, states)...))

function print(io::IO, state::FockState) 
    print(io, "[")
    for (k, n) in pairs(state)
        for i in 1:n
            print(io, " ", k)
        end
    end
    print(io, " ]")
end

==(state1::FockState, state2::FockState) = all(n1 == n2 for (k, (n1, n2)) in iter_all(state1, state2))

struct DictFockState{K <: Signed, N <: Unsigned, D <: AbstractDict{K, N}} <: FockState{K, N}
    occupation_numbers::D
end

DictFockState{K, N}(occupation_numbers::Pair{K, N}...) where {K <: Signed, N <: Unsigned} = DictFockState(Dict(occupation_numbers))

function DictFockState(occupation_numbers::Pair{K, N}...) where {K, N}
    DictFockState(Dict(signed(k) => force_unsigned(n) for (k, n) in occupation_numbers))
end

frozen(state::DictFockState) = DictFockState(ImmutableDict(state.occupation_numbers...))

pairs(state::DictFockState) = pairs(state.occupation_numbers)
iterate(state::DictFockState) = iterate(pairs(state))
iterate(state::DictFockState, progress) = iterate(pairs(state), progress)
length(state::DictFockState) = length(state.occupation_numbers)

relevant_momenta(state::DictFockState) = keys(state.occupation_numbers)

function occupation_number(state::DictFockState{K, N}, momentum::K) where {K, N}
    get(state.occupation_numbers, momentum, zero(N))
end

function add_particle!(state::DictFockState{K, N}, momentum::K) where {K, N}
    state.occupation_numbers[momentum] = occupation_number(state, momentum) + one(N)
end

function x_flipped(state::DictFockState{K, N, D}) where {K, N, D}
    DictFockState{K, N, D}(D(-k => n for (k, n) in state))
end

x_parity(state::DictFockState) = state == x_flipped ? Even : nothing