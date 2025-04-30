export generate_states

using .IntPartitions

import Base: isvalid

function generate_states(subspace::SubSpace)
    (net_pos_k, net_neg_k) = sign_split(subspace.eigenspace.momentum)
    max_k_int = max_momentum(subspace) - abs(subspace.eigenspace.momentum)

    states = flatmap(0:max_k_int+1) do k_internal
        gross_momentum_states(subspace, k_internal + net_pos_k, k_internal + net_neg_k)
    end
    
    # Zeros have already been filtered
    symmetrise_states(states, subspace.eigenspace.x_symmetrisation, false)
end

function max_momentum(subspace::SubSpace)
    floor(Unsigned, sqrt((subspace.max_energy^2 - 1)) / k_unit(subspace.fock_space))
end

function gross_momentum_states(subspace::SubSpace, pos_momentum, neg_momentum)
    pos_momentum_splits = split_momentum(subspace, pos_momentum)

    flatmap(pos_momentum_splits) do pos_split
        neg_momentum_splits = split_momentum(subspace.fock_space, pos_split.remaining_energy, neg_momentum)

        filtered_neg_momentum_splits = nondegen_splits(neg_momentum_splits, pos_momentum == neg_momentum, pos_split.momenta, subspace.eigenspace.x_symmetrisation)

        flatmap(filtered_neg_momentum_splits) do neg_split
            states_with_stationary(subspace.eigenspace, neg_split.remaining_energy, pos_split.momenta, neg_split.momenta)
        end
    end
end

struct MomentumSplit{K <: Unsigned, E}
    remaining_energy::E
    momenta::Vector{K}
end
MomentumSplit(space::BoundedFockSpace, available_energy::E, momenta::Vector{K}) where {K, E} = MomentumSplit{K, E}(available_energy - free_energy(space, momenta), momenta)

isvalid(split::MomentumSplit) = split.remaining_energy > 0

function split_momentum(space::BoundedFockSpace, max_energy, gross_momentum)
    takewhile(isvalid, (MomentumSplit(space, max_energy, momenta) for momenta in energy_ordered_partitions(gross_momentum)))
end

split_momentum(subspace::SubSpace, gross_momentum) = split_momentum(subspace.fock_space, subspace.max_energy, gross_momentum)

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