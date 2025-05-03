export BoundedFockSpace, FockSpaceImpl, k_unit, free_energy, hamiltonian, FreeHamiltonian

import Base.broadcastable

using Base.Iterators
using .IntPartitions

abstract type BoundedFockSpace{E <: AbstractFloat} end

Base.broadcastable(space::BoundedFockSpace) = Ref(space)

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
    free_energy(space, representative_fockstate(state))
end

function free_energy(space::BoundedFockSpace{E}, momenta)::E where E
    sum((free_energy(space, k) for k in momenta); init = zero(E))
end

struct FreeHamiltonian{E, F <: BoundedFockSpace{E}} <: NiceMatrix{E}
    space::F
end

hamiltonian(space::BoundedFockSpace) = FreeHamiltonian(space)

diagonal_element(hamiltonian::FreeHamiltonian, state::FockState) = free_energy(hamiltonian.space, state)