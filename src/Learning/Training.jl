export learn_components

using ..Hamiltonian

using Plots, Measurements
using Lux, Random, Optimisers, Zygote

function learn_components(model, fock_space::Phi4Space, eigenspace::EigenSpace, energy_levels, n_components)
    subhams = sub_hamiltonians(fock_space, eigenspace, energy_levels...)

    rng = Random.default_rng()
    Random.seed!(rng, 0)
    device = gpu_device()

    optimiser = Adam(0.001f0)

    weights, lux_state = Lux.setup(rng, model) |> device
    train_state = Training.TrainState(model, weights, lux_state, optimiser)

    plt = plot(0)
    ylabel!(plt, "log(loss)")
    xlabel!(plt, "#Datapoints Learned")

    data = Float64[]

    for (E_max, (states, H)) in zip(energy_levels, subhams)
        eigstates, eigenergies = spectrum(H, n_components)
        
        for i in 1:(length(states) / 5)
            grads, loss, stats, train_state = Training.single_train_step!(AutoZygote(), MSELoss(), ((states, context_vec(fock_space, E_max)), eigstates), train_state)
            push!(data, log(loss))
            plot(plt, data)
        end
    end    
end