export BoundedFockSpace, FockSpaceImpl, k_unit, free_energy

using Base.Iterators
using .IntPartitions

abstract type BoundedFockSpace{E <: AbstractFloat} end

struct FockSpaceImpl{E <: AbstractFloat} <: BoundedFockSpace{E}
    k_unit::E
end

FockSpaceImpl(k_unit::Real) = FockSpaceImpl(float(k_unit))

k_unit(space::FockSpaceImpl) = space.k_unit

function free_energy(space::BoundedFockSpace{E}, momentum::Integer)::E where E
    sqrt(1 + (k_unit(space) * momentum)^2)
end

function free_energy(space::BoundedFockSpace{E}, state::FockState)::E where E
    sum((n * free_energy(space, k) for (k, n) in state); init = zero(E))
end

function free_energy(space::BoundedFockSpace{E}, state::SymmetrisedFockState)::E where E
    free_energy(space, state.base_state)
end

function free_energy(space::BoundedFockSpace{E}, momenta)::E where E
    sum((free_energy(space, k) for k in momenta); init = zero(E))
end