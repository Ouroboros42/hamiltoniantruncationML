export FockSpace, k_unit, free_energy, hamiltonian, FreeHamiltonian

import Base: broadcastable, size

twoPi(Ftype) = convert(Ftype, 2π)
angular_inverse(T, x)::T = twoPi(T) / convert(T, x)
angular_inverse(x) = angular_inverse(typeof(x), x)

abstract type FockSpace{E <: AbstractFloat} end

Base.broadcastable(space::FockSpace) = Ref(space)
size(space::FockSpace) = angular_inverse(k_unit(space))

struct FockSpaceImpl{E <: AbstractFloat} <: FockSpace{E}
    k_unit::E
end
k_unit(space::FockSpaceImpl) = space.k_unit

FockSpace{E}(; k_unit) where E = FockSpaceImpl{E}(convert(E, k_unit))
function FockSpace(; k_unit)
    k_unit = float(k_unit)
    FockSpaceImpl{typeof(k_unit)}(k_unit)
end

FockSpace{E}(size) where E = FockSpaceImpl{E}(angular_inverse(E, size))
function FockSpace(size)
    size = float(size)
    FockSpaceImpl{typeof(size)}(angular_inverse(size))
end

function free_energy(space::FockSpace{E}, momentum::Integer)::E where E
    sqrt(1 + (k_unit(space) * momentum)^2)
end

function free_energy(space::FockSpace{E}, state::FockState)::E where E
    sum((n * free_energy(space, k) for (k, n) in state); init = zero(E))
end

function free_energy(space::FockSpace{E}, state::SymmetrisedFockState)::E where E
    free_energy(space, representative_fockstate(state))
end

function free_energy(space::FockSpace{E}, momenta)::E where E
    sum((free_energy(space, k) for k in momenta); init = zero(E))
end

struct FreeHamiltonian{E, F <: FockSpace{E}} <: NiceMatrix{E}
    space::F
end

hamiltonian(space::FockSpace) = FreeHamiltonian(space)

diagonal_element(hamiltonian::FreeHamiltonian, state::FockState) = free_energy(hamiltonian.space, state)