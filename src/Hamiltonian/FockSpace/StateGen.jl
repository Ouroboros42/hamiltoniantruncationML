export generate_states

using .IntPartitions

import Base: isvalid

function generate_states(fockspace::BoundedFockSpace, eigenspace::EigenSpace, max_energy)
    (net_pos_k, net_neg_k) = sign_split(eigenspace.momentum)
    max_k_int = max_momentum(fockspace, max_energy) - abs(eigenspace.momentum)

    states = flatmap(0:max_k_int+1) do k_internal
        gross_momentum_states(fockspace, eigenspace, max_energy, k_internal + net_pos_k, k_internal + net_neg_k)
    end
    
    # Zeros have already been filtered
    symmetrise_states(states, eigenspace.x_symmetrisation, false)
end

function max_momentum(fockspace::BoundedFockSpace, max_energy)
    floor(Unsigned, sqrt(max_energy^2 - 1) / k_unit(fockspace))
end

function gross_momentum_states(fockspace::BoundedFockSpace, eigenspace::EigenSpace, max_energy, pos_momentum, neg_momentum)
    pos_momentum_splits = split_momentum(fockspace, max_energy, pos_momentum)

    flatmap(pos_momentum_splits) do pos_split
        neg_momentum_splits = split_momentum(fockspace, pos_split.remaining_energy, neg_momentum)

        filtered_neg_momentum_splits = nondegen_splits(neg_momentum_splits, pos_momentum == neg_momentum, pos_split.momenta, eigenspace.x_symmetrisation)

        flatmap(filtered_neg_momentum_splits) do neg_split
            states_with_stationary(eigenspace, neg_split.remaining_energy, pos_split.momenta, neg_split.momenta)
        end
    end
end

struct MomentumSplit{K <: Unsigned, E}
    remaining_energy::E
    momenta::Vector{K}
end

function MomentumSplit(space::BoundedFockSpace{E}, available_energy, momenta::Vector{K}) where {K, E}
    MomentumSplit{K, E}(convert(E, available_energy) - free_energy(space, momenta), momenta)
end

isvalid(split::MomentumSplit) = split.remaining_energy > 0

function split_momentum(space::BoundedFockSpace, max_energy, gross_momentum)
    takewhile(isvalid, (MomentumSplit(space, max_energy, momenta) for momenta in energy_ordered_partitions(gross_momentum)))
end

nondegen_splits(momentum_splits, filter_needed::Bool, lexicographic_bound::Vector{K}, x_symmetrisation::Nothing) where {K <: Unsigned} = momentum_splits
function nondegen_splits(momentum_splits, filter_needed::Bool, lexicographic_max::Vector{K}, x_symmetrisation::Parity) where {K <: Unsigned}
    bound_func = @match x_symmetrisation begin
        &Even => <=
        &Odd => <
    end

    Iterators.filter(momentum_splits) do (; momenta)
        !filter_needed || bound_func(momenta, lexicographic_max)
    end
end

function states_with_stationary(eigenspace::EigenSpace{K, N}, remaining_energy, pos_momenta, neg_momenta) where {K, N}
    n_stationary_parity = shift(eigenspace.n_parity, length(pos_momenta) + length(neg_momenta))
    max_n_stationary = floor(N, remaining_energy)

    (make_state(K, N, pos_momenta, neg_momenta, n_stationary) for n_stationary in range(n_stationary_parity, max_n_stationary))
end

function make_state(::Type{K}, ::Type{N}, pos_momenta, neg_momenta, n_stationary) where {K <: Signed, N <: Unsigned}
    counts = DictFockState{K, N}(0 => n_stationary)

    for k in pos_momenta
        add_particle!(counts, convert(K, k))
    end

    for k in neg_momenta
        add_particle!(counts, -convert(K, k))
    end

    counts
end