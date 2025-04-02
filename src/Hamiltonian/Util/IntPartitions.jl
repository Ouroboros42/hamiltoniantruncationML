module IntPartitions

export energy_ordered_partitions

import ..Hamiltonian: force_unsigned

import Base: iterate, IteratorSize, last, isempty

using DataStructures

struct PartitionBuilder{N <: Unsigned}
    total::N
end

"""Alternative to Combinatorics.partitions which is ordered according to decreasing relativistic energy for a set of particles with those momenta."""
energy_ordered_partitions(total::N) where {N <: Integer} = PartitionBuilder(force_unsigned(total))

struct PartialPartition{N <: Unsigned}
    sequence::Vector{N}
    remaining_total::N
end

last(partial::PartialPartition) = last(partial.sequence)

start_partition(first::N, target::N) where {N <: Unsigned} = PartialPartition([first], target - first)

function extended_by(partition::PartialPartition{N}, next::N) where {N <: Unsigned}
    PartialPartition(finish_with(partition, next), partition.remaining_total - next)
end

function finish_with(partition::PartialPartition{N}, last::N) where {N <: Unsigned}
    vcat(partition.sequence, [last])
end

const PartitionProgress{N <: Unsigned} = Deque{PartialPartition{N}}

IteratorSize(::PartitionBuilder) = Base.SizeUnknown()

isempty(builder::PartitionBuilder) = isempty(builder.partials)

function iterate(builder::PartitionBuilder{N}) where {N <: Unsigned}
    if iszero(builder.total)
        return (Vector{N}(), nothing)
    end

    partials = PartitionProgress{N}()

    for first in 0x1:(builder.total - 0x1)
        push!(partials, start_partition(first, builder.total))
    end
    
    return ([builder.total], partials)
end

function iterate(builder::PartitionBuilder{N}, ::Nothing) where {N <: Unsigned}
    nothing
end

function iterate(builder::PartitionBuilder{N}, state::PartitionProgress{N}) where {N <: Unsigned}
    while !isempty(state)
        partial = popfirst!(state)

        for next_element in 0x1:last(partial)
            if next_element == partial.remaining_total
                return (finish_with(partial, next_element), state)
            end
            
            extended = extended_by(partial, next_element)
            push!(state, extended)
        end
    end

    nothing
end

end