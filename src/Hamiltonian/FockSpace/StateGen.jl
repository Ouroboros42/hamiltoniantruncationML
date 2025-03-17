export generate_states

function generate_states(space::BoundedFockSpace{E}, max_energy::Real; x_symmetrisation::MaybeParity = nothing, kwargs...) where { E <: AbstractFloat }
    symmetrise_states(generate_fockstates(space, convert(E, max_energy); kwargs...), x_symmetrisation)
end

function generate_fockstates(space::BoundedFockSpace{E}, max_energy::E; momentum::K = 0, n_parity::MaybeParity = nothing) where {E <: AbstractFloat, K <: Signed}
    abs_k = abs(momentum)
    max_k = max_momentum(space, max_energy)

    (net_pos_k, net_neg_k) = momentum >= 0 ? (abs_k, 0) : (0, abs_k)
    max_k_int = max_k - abs_k

    flatmap_until_empty(0:max_k_int+1) do k_internal
        gross_momentum_states(space, max_energy, unsigned(k_internal + net_pos_k), unsigned(k_internal + net_neg_k), n_parity)
    end
end

flatmap_until_empty(f, iter) = flatten(takewhile(!isempty, Iterators.map(f, iter)))

function max_momentum(space::BoundedFockSpace{E}, max_energy::E) where {E <: AbstractFloat}
    floor(Integer, sqrt((max_energy^2 - 1)) / k_unit(space))
end

function gross_momentum_states(space::BoundedFockSpace, max_energy::AbstractFloat, pos_momentum::K, neg_momentum::K, n_parity::MaybeParity) where {K <: Unsigned}
    flatmap(momentum_splits(space, max_energy, pos_momentum)) do (energy_minus_pos, pos_momenta)
        flatmap(momentum_splits(space, energy_minus_pos, neg_momentum)) do (energy_minus_neg, neg_momenta)
            states_with_stationary(energy_minus_neg, pos_momenta, neg_momenta, n_parity)
        end
    end
end

function momentum_splits(space::BoundedFockSpace, max_energy::AbstractFloat, gross_momentum::K) where {K <: Unsigned}
    energies_and_momenta = (
        (max_energy - free_energy(space, pos_momenta), pos_momenta) for pos_momenta in PartitionBuilder(gross_momentum) 
    )
    
    takewhile(energies_and_momenta) do (remaining_energy, _)
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