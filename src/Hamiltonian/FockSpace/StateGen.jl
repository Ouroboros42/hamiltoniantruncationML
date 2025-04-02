export generate_states

using .IntPartitions

generate_states(space::BoundedFockSpace{E}, max_energy; kwargs...) where E = generate_states(space, convert(E, max_energy); kwargs...) 
function generate_states(space::BoundedFockSpace{E}, max_energy::E; momentum::K = 0, n_parity::MaybeParity = nothing, x_symmetrisation::MaybeParity = nothing) where {E, K <: Signed}
    abs_k = abs(momentum)
    max_k = max_momentum(space, max_energy)

    (net_pos_k, net_neg_k) = momentum >= 0 ? (abs_k, 0) : (0, abs_k)
    max_k_int = max_k - abs_k

    states = flatmap(0:max_k_int+1) do k_internal
        gross_momentum_states(space, max_energy, unsigned(k_internal + net_pos_k), unsigned(k_internal + net_neg_k), n_parity, x_symmetrisation)
    end
    
    # Zeros have already been filtered
    symmetrise_states(states, x_symmetrisation, false)
end

function max_momentum(space::BoundedFockSpace{E}, max_energy::E) where E
    floor(Integer, sqrt((max_energy^2 - 1)) / k_unit(space))
end

function gross_momentum_states(space::BoundedFockSpace, max_energy::AbstractFloat, pos_momentum::K, neg_momentum::K, n_parity::MaybeParity, x_symmetrisation::MaybeParity) where {K <: Unsigned}
    pos_momentum_splits = split_momentum(space, max_energy, pos_momentum)

    flatmap(pos_momentum_splits) do (energy_minus_pos, pos_momenta)
        neg_momentum_splits = cutoff_split_momenta(split_momentum(space, energy_minus_pos, neg_momentum), pos_momentum == neg_momentum, pos_momenta, x_symmetrisation)

        flatmap(neg_momentum_splits) do (energy_minus_neg, neg_momenta)
            states_with_stationary(energy_minus_neg, pos_momenta, neg_momenta, n_parity)
        end
    end
end

cutoff_split_momenta(momentum_splits, filter_needed::Bool, lexicographic_bound::Vector{K}, x_symmetrisation::Nothing) where {K <: Unsigned} = momentum_splits

function cutoff_split_momenta(momentum_splits, filter_needed::Bool, lexicographic_max::Vector{K}, x_symmetrisation::Parity) where {K <: Unsigned}
    bound_func = @match x_symmetrisation begin
        &Even => <=
        &Odd => <
    end

    Iterators.filter(momentum_splits) do (; momenta)
        !filter_needed || bound_func(momenta, lexicographic_max)
    end 
end

function split_momentum(space::BoundedFockSpace, max_energy::AbstractFloat, gross_momentum::K) where {K <: Unsigned}
    energies_and_momenta = Iterators.map(energy_ordered_partitions(gross_momentum)) do momenta
        (; remaining_energy = max_energy - free_energy(space, momenta), momenta)
    end

    takewhile(energies_and_momenta) do (; remaining_energy)
        remaining_energy > 0
    end
end

function states_with_stationary(remaining_energy::AbstractFloat, pos_momenta::Vector{K}, neg_momenta::Vector{K}, n_parity::MaybeParity) where {K <: Unsigned}
    n_stationary_parity = shift(n_parity, length(pos_momenta) + length(neg_momenta))
    max_n_stationary = floor(Unsigned, remaining_energy)

    (make_state(pos_momenta, neg_momenta, n_stationary) for n_stationary in range(n_stationary_parity, max_n_stationary))
end

function make_state(pos_momenta::Vector{K}, neg_momenta::Vector{K}, n_stationary::N) where {K <: Unsigned, N <: Unsigned}
    counts = DictFockState(Dict(zero(signed(K)) => n_stationary))

    for k in pos_momenta
        add_particle!(counts, signed(k))
    end

    for k in neg_momenta
        add_particle!(counts, -signed(k))
    end

    counts
end