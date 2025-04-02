export Parity, MaybeParity, Even, Odd

import Base: range, Set

@enum Parity::UInt8 begin
    Even = 0x0
    Odd = 0x1
end

const MaybeParity = Union{Parity, Nothing}

const ALL_DEFINITE_PARITIES = Set{Parity}([Even, Odd])
const ALL_PARITIES = Set{MaybeParity}([nothing, Even, Odd])

Set(::Type{Parity}) = ALL_DEFINITE_PARITIES
Set(::Type{MaybeParity}) = ALL_PARITIES

number_parity(n::Unsigned) = Parity(n % 2)

shift(parity::Nothing, ::Integer) = parity
shift(parity::Parity, n::Integer) = number_parity(Unsigned(parity) + Unsigned(n))

range(parity::Parity, stop::Unsigned) = Unsigned(parity):0x2:stop
range(parity::Nothing, stop::Unsigned) = 0x0:stop