export EigenSpace, SubSpace

import Base: convert, show, print

struct EigenSpace{K <: Signed, N <: Unsigned, XP <: MaybeParity, NP <: MaybeParity}
    momentum::K
    x_symmetrisation::XP
    n_parity::NP
end

EigenSpace{K, N}(momentum, x_symmetrisation::XP, n_parity::NP = nothing) where {K, N, XP, NP} = EigenSpace{K, N, XP, NP}(convert(K, momentum), x_symmetrisation, n_parity)
EigenSpace{K, N}(momentum = 0; x_symmetrisation = nothing, n_parity = nothing) where {K, N} = EigenSpace{K, N}(momentum, x_symmetrisation, n_parity)

EigenSpace(momentum::K, args...; kwargs...) where K = EigenSpace{signed(K), unsigned(K)}(momentum, args...; kwargs...)

print(io::IO, eigenspace::EigenSpace) = print(io, "K=$(eigenspace.momentum) X-$(parity_string(eigenspace.x_symmetrisation)) N-$(parity_string(eigenspace.n_parity))")
show(io::IO, mime::MIME"text/plain", eigenspace::EigenSpace) = print(io, "$(typeof(eigenspace)): $(eigenspace)")

struct SubSpace{E, S <: BoundedFockSpace{E}, Eig <: EigenSpace}
    fock_space::S
    eigenspace::Eig
    max_energy::E
end

function show(io::IO, ::MIME"text/plain", subspace::SubSpace)
    print(io, "Subspace of $(subspace.fock_space)\n  E < $(subspace.max_energy)\n  $(subspace.eigenspace)")
end

function print(io::IO, subspace::SubSpace)
    print(io, "$(subspace.fock_space) | E<$(subspace.max_energy) $(subspace.eigenspace)")
end

SubSpace(fock_space::S, eigenspace::Eig, max_energy) where {E, S <: BoundedFockSpace{E}, Eig <: EigenSpace} = SubSpace{E, S, Eig}(fock_space, eigenspace, convert(E, max_energy))
