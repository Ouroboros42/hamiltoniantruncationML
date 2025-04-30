export Parity, MaybeParity, Even, Odd, parity_string

import Base: range, Set, convert

@enum Parity::UInt8 begin
    Even = 0x0
    Odd = 0x1
end

const MaybeParity = Union{Parity, Nothing}

const ALL_DEFINITE_PARITIES = Set{Parity}([Even, Odd])
const ALL_PARITIES = Set{MaybeParity}([nothing, Even, Odd])

parity_string(parity::Nothing) = "all"
parity_string(parity::Parity) = string(parity)

Set(::Type{Parity}) = ALL_DEFINITE_PARITIES
Set(::Type{MaybeParity}) = ALL_PARITIES

number_parity(n::Unsigned) = Parity(n % 2)

shift(parity::Nothing, ::Integer) = parity
shift(parity::Parity, n::Integer) = number_parity(Unsigned(parity) + Unsigned(n))

convert(::Type{U}, parity::Parity) where {U <: Unsigned} = convert(U, Unsigned(parity))

range(parity::Parity, stop::N) where N = convert(N, parity):convert(N, 2):stop
range(parity::Nothing, stop) = zero(stop):stop