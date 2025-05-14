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

make_context(space::FockSpace, coupling, max_energy) = [ k_unit(space); coupling; max_energy ]

i_wrap(collection, i) = (i - 1) % length(collection) + 1

function learn_components!(train_states, fockspace::FockSpace, eigenspace::EigenSpace, max_energies, couplings, n_components, n_epochs;
    plot_kwargs = (; ),
    loss_plot_kwargs = (; ),
    corr_plot_kwargs = (; ),
    backend = AutoZygote(), lossfunc = MSELoss()    
)
    solved_subhams = map(sub_hamiltonians(fockspace, eigenspace, max_energies, couplings)) do subspace
        (; subspace...,
            context = make_context(fockspace, subspace.coupling, subspace.max_energy),
            components = normalise_components(spectrum(subspace.hamiltonian, n_components))
        )
    end

    n_subspaces = length(solved_subhams)
    n_training_steps = n_epochs * n_subspaces

    loss_history = fill(NaN, (n_training_steps, length(train_states)))

    for i_epoch in 1:n_epochs
        display_index = i_wrap(solved_subhams, i_epoch)

        for (i_subspace, (; states, context, components)) in enumerate(solved_subhams)
            i_training_step = n_subspaces * (i_epoch - 1) + i_subspace

            losses = map(train_states) do train_state
                grads, loss, stats, train_state = Training.single_train_step!(backend, lossfunc, ((context, states), components), train_state)
                loss
            end            
            
            loss_history[i_training_step, :] = log.(permutedims(losses))

            if display_index == i_subspace
                plot(loss_history; plot_kwargs..., loss_plot_kwargs...)
            end
        end

        @info "Completed training epoch $i_epoch"
    end

    (; loss_history, train_states)
end