export learn_components!, apply, setup_model, make_context, state_scorer, sort_by_score, normalise_components

import Lux.apply, Lux.Training.TrainState

apply(trainstate::TrainState, context, states) = apply(trainstate, (context, states))
apply(trainstate::TrainState, (context, state)::Tuple{Any, FieldState}) = only(apply(trainstate, (context, [state])))
function apply(trainstate::TrainState, x)
    output, state = apply(trainstate.model, x, trainstate.parameters, trainstate.states)
    output
end

normalise_components(components) = abs.(components)
normalise_components((components, _)::Tuple) = normalise_components(components)

function setup_model(model)
    rng = Random.default_rng()
    Random.seed!(rng, 0)
    device = cpu_device()

    optimiser = Adam(0.001f0)

    weights, lux_state = Lux.setup(rng, model) |> device
    
    TrainState(model, weights, lux_state, optimiser)
end

make_context(space::FockSpace, coupling, max_energy) = Float32[ k_unit(space); coupling; max_energy ]

i_wrap(collection, i) = (i - 1) % length(collection) + 1

function learn_components!(train_states, fockspace::FockSpace, eigenspace::EigenSpace, max_energies, couplings, n_components, n_epochs;
    plot_kwargs = (; ),
    baseline_predictors::Vector = [],
    backend = AutoZygote(), lossfunc = MSELoss()    
)
    solved_subhams = map(sub_hamiltonians(fockspace, eigenspace, max_energies, couplings)) do subspace
        (; subspace...,
            context = make_context(fockspace, subspace.coupling, subspace.max_energy),
            components = normalise_components(spectrum(subspace.hamiltonian, n_components))
        )
    end
    
    loss_history = fill(NaN, (n_epochs, length(train_states) + length(baseline_predictors)))

    for i_epoch in 1:n_epochs
        display_index = i_wrap(solved_subhams, i_epoch)

        total_losses = mapreduce(.+, solved_subhams) do (; states, context, components)
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
                    
        loss_history[i_epoch, :] = log.(permutedims(total_losses ./ length(solved_subhams)))

        plot(loss_history; plot_kwargs...)

        @info "Completed training epoch $i_epoch"
    end

    (; loss_history, train_states)
end