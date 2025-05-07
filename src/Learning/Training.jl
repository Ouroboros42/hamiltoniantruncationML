export learn_components!, apply, setup_model, make_context, state_scorer, getstates, getscores

using ..Hamiltonian

using Plots, Measurements, StatsPlots
using Lux, Random, Optimisers, Zygote

import Lux.apply, Lux.Training.TrainState

apply(trainstate::TrainState, context, states) = apply(trainstate, (context, states))
apply(trainstate::TrainState, (context, state)::Tuple{Any, FieldState}) = only(apply(trainstate, (context, [state])))
function apply(trainstate::TrainState, x)
    output, state = apply(trainstate.model, x, trainstate.parameters, trainstate.states)
    output
end

struct ScoredStates{S, F}
    states::S
    scores::F
end

getstates(scored::ScoredStates) = scored.states
getscores(scored::ScoredStates) = scored.scores

state_scorer(trainstate::TrainState, context) = states -> ScoredStates(states, apply(trainstate, (context, states)))
state_scorer(trainstate::TrainState, args...) = state_scorer(trainstate, make_context(args...))

normalise_components(components) = abs.(components)
normalise_components((components, _)::Tuple) = normalise_components(components)

function setup_model(model)
    rng = Random.default_rng()
    Random.seed!(rng, 0)
    device = gpu_device()

    optimiser = Adam(0.001f0)

    weights, lux_state = Lux.setup(rng, model) |> device
    
    TrainState(model, weights, lux_state, optimiser)
end

make_context(space::FockSpace, coupling, max_energy) = [ k_unit(space); coupling; max_energy ]

i_wrap(collection, i) = (i - 1) % length(collection) + 1

function learn_components!(train_state, fockspace::FockSpace, eigenspace::EigenSpace, max_energies, couplings, n_components, n_epochs;
    backend = AutoZygote(), lossfunc = MSELoss()    
)
    data = Float64[]

    solved_subhams = map(sub_hamiltonians(fockspace, eigenspace, max_energies, couplings)) do subspace
        (;
            subspace...,
            context = make_context(fockspace, subspace.coupling, subspace.max_energy),
            components = normalise_components(spectrum(subspace.hamiltonian, n_components))
        )
    end

    for i_epoch in 1:n_epochs
        display_index = i_wrap(solved_subhams, i_epoch)

        for (i_subspace, (; coupling, states, context, components)) in enumerate(solved_subhams)
            grads, loss, stats, train_state = Training.single_train_step!(backend, lossfunc, ((context, states), components), train_state)
            
            push!(data, log(loss))

            if display_index == i_subspace
                predicted_components = apply(train_state, context, states)

                training = plot(data, xlabel="#Training Steps", ylabel="log(loss)", legend=false)

                side_by_sides = map(1:n_components, eachcol(components), eachcol(predicted_components)) do i, actual, prediction
                    groupedbar(hcat(actual, prediction), title = "State $i Components, g=$coupling", label = ["True" "Prediction"])
                end

                plot(training, side_by_sides..., layout=(n_components + 1, 1), size=(1300, 700))
            end
        end

        @info "Completed training epoch $i_epoch"
    end        

    train_state
end