export StatsAccumulator, mean, total, count, variance, stdev, samplevar, samplestdev, total_measure, mean_measure

import Base: push!

using Measurements
import Measurements: measurement

mutable struct StatsAccumulator{T}
    count::UInt
    total::T
    sumsquares::T
end

StatsAccumulator(init::T) where T = StatsAccumulator{T}(1, init, init^2)
StatsAccumulator(::Type{T}) where T = StatsAccumulator(zero(T))

function push!(acc::StatsAccumulator{T}, data::T) where T
    acc.count += 1
    acc.total += data
    acc.sumsquares += data ^ 2
end


mean(seq) = sum(seq) / length(seq)

total(acc::StatsAccumulator) = acc.total
count(acc::StatsAccumulator) = acc.count
mean(acc::StatsAccumulator) = total(acc) / count(acc)
variance(acc::StatsAccumulator) = (acc.sumsquares / acc.count) - (mean(acc) ^ 2)
stdev(acc) = sqrt(variance(acc))

samplevar(acc::StatsAccumulator) = variance(acc) * acc.count / (acc.count - 1)
samplestdev(acc) = sqrt(samplevar(acc))

mean_measure(acc::StatsAccumulator) = mean(acc) ± samplestdev(acc)
total_measure(acc::StatsAccumulator) = total(acc) ± (samplestdev(acc) * count(acc))