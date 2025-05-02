export learn_components

using ..Hamiltonian

using Plots, Measurements, StatsPlots
using Lux, Random, Optimisers, Zygote

function __init__()
    gr(show = true)
end

function normalise_components(components)
    abs.(components)
end

function learn_components(model, fock_space::Phi4Space, eigenspace::EigenSpace, energy_levels, n_components)
    subhams = sub_hamiltonians(fock_space, eigenspace, energy_levels...)

    rng = Random.default_rng()
    Random.seed!(rng, 0)
    device = gpu_device()

    optimiser = Adam(0.001f0)

    weights, lux_state = Lux.setup(rng, model) |> device
    train_state = Training.TrainState(model, weights, lux_state, optimiser)

    data = Float64[]

    for (E_max, (states, H)) in zip(energy_levels, subhams)
        context = context_vec(fock_space, E_max)

        eigstates, eigenergies = spectrum(H, n_components)
        predicted_components, lux_state = Lux.apply(model, (states, context), weights, lux_state)

        components = normalise_components(eigstates)

        for i in 1:1000
            grads, loss, stats, train_state = Training.single_train_step!(AutoZygote(), MSELoss(), ((states, context), components), train_state)
            
            push!(data, log(loss))
        end

        training = plot(data, xlabel="#Training Steps", ylabel="log(loss)", legend=false)

        side_by_sides = map(1:n_components, eachcol(components), eachcol(predicted_components)) do i, actual, prediction
            groupedbar(hcat(actual, prediction), title = "State $i Components", label = ["True" "Prediction"])
        end

        plot(training, side_by_sides..., layout=(n_components + 1, 1), size=(1300, 700))
    end    
end