export learn_components!, apply, setup_model, restart_model, make_context, state_scorer, sort_by_score, normalise_components

import Lux.apply, Lux.Training.TrainState

apply(trainstate::TrainState, context, states) = apply(trainstate, (context, states))
apply(trainstate::TrainState, (context, state)::Tuple{Any, FieldState}) = only(apply(trainstate, (context, [state])))
function apply(trainstate::TrainState, x)
    output, state = apply(trainstate.model, x, trainstate.parameters, trainstate.states)
    output
end

normalise_components(components) = abs.(components)
normalise_components((components, _)::Tuple) = normalise_components(components)

function setup_model(model, adam_params... = 0.01f0)
    rng = Random.default_rng()
    Random.seed!(rng, 0)
    device = cpu_device()

    optimiser = Adam(adam_params...)

    weights, lux_state = Lux.setup(rng, model) |> device
    
    TrainState(model, weights, lux_state, optimiser)
end

restart_model(model, params) = TrainState(model, params..., Adam(0.01f0))

function make_context(space::FockSpace, coupling, max_energy; coupling_in_context::Bool = true, cutoff_in_context::Bool = true)
    context = Float32[ k_unit(space) ]

    if coupling_in_context
        context = Float32[ context; coupling ]
    end

    if cutoff_in_context
        context = Float32[ context; max_energy ]
    end

    context
end

function learn_components!(train_states, fockspace::FockSpace, eigenspace::EigenSpace, max_energies, couplings, n_components, n_epochs;
    plot_kwargs = (; ),
    baseline_predictors::Vector = [],
    backend = AutoZygote(), lossfunc = MSELoss(),
    coupling_in_context::Bool=true,
    cutoff_in_context::Bool=true
)
    subhams = sub_hamiltonians(fockspace, eigenspace, max_energies, couplings)

    solutions = map(subhams) do subspace
        (;
            context = make_context(fockspace, subspace.coupling, subspace.max_energy; coupling_in_context, cutoff_in_context),
            components = normalise_components(spectrum(subspace.hamiltonian, n_components))
        )
    end

    loss_history = fill(NaN, (n_epochs, length(train_states) + length(baseline_predictors)))
    
    subspace_indices = collect(enumerate(eachindex(subhams)))

    for i_epoch in 1:n_epochs
        subspace_indices = shuffle!(subspace_indices)

        total_losses = mapreduce(.+, merge(subhams[bi], solutions[i]) for (i, bi) in subspace_indices) do (; states, context, components)
            trained_states = states[2:end] # Don't train vacuum
            trained_components = components[2:end]

            baseline_losses = map(baseline_predictors) do predictor
                lossfunc(predictor(trained_components), trained_components)
            end

            network_losses = map(train_states) do train_state
                grads, loss, stats, train_state = Training.single_train_step!(backend, lossfunc, ((context, trained_states), trained_components), train_state)
                loss
            end            

            vcat(baseline_losses, network_losses)
        end
                    
        loss_history[i_epoch, :] = log.(permutedims(total_losses ./ length(total_losses)))

        plot(loss_history; plot_kwargs...)

        @info "Completed training epoch $i_epoch"
    end

    parameters = map(train_states) do train_state
        train_state.parameters, train_state.states
    end

    (; loss_history,  parameters, subspaces = subhams)
end