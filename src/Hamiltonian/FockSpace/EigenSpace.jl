export EigenSpace, DEFAULT_EIGENSPACE

import Base: show, print

struct EigenSpace{K <: Signed, N <: Unsigned, XP <: MaybeParity, NP <: MaybeParity}
    momentum::K
    x_symmetrisation::XP
    n_parity::NP
end

EigenSpace{K, N}(momentum, x_symmetrisation::XP, n_parity::NP = nothing) where {K, N, XP, NP} = EigenSpace{K, N, XP, NP}(convert(K, momentum), x_symmetrisation, n_parity)
EigenSpace{K, N}(momentum = 0; x_symmetrisation = nothing, n_parity = nothing) where {K, N} = EigenSpace{K, N}(momentum, x_symmetrisation, n_parity)

EigenSpace(momentum::K = 0, args...; kwargs...) where K = EigenSpace{signed(K), unsigned(K)}(momentum, args...; kwargs...)

print(io::IO, eigenspace::EigenSpace) = print(io, "K=$(eigenspace.momentum) X-$(parity_string(eigenspace.x_symmetrisation)) N-$(parity_string(eigenspace.n_parity))")
show(io::IO, mime::MIME"text/plain", eigenspace::EigenSpace) = print(io, "$(typeof(eigenspace)): $(eigenspace)")

const DEFAULT_EIGENSPACE = EigenSpace{Int8, UInt8}(0, Even, Even)