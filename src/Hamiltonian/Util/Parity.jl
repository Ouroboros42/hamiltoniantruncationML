export Parity, MaybeParity, Even, Odd, AllParities

import Base: range

@enum Parity::UInt8 begin
    Even = 0x0
    Odd = 0x1
end

const MaybeParity = Union{Parity, Nothing}

const AllParities = Set{MaybeParity}([nothing, Even, Odd])

number_parity(n::Unsigned) = Parity(n % 2)

shift(parity::Nothing, ::Integer) = parity
shift(parity::Parity, n::Integer) = number_parity(Unsigned(parity) + Unsigned(n))

range(parity::Parity, stop::Unsigned) = Unsigned(parity):0x2:stop
range(parity::Nothing, stop::Unsigned) = 0x0:stop