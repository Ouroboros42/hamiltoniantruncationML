export learn_components

using ..Hamiltonian

using Plots, Measurements
using Lux, Random, Optimisers, Zygote

recursive_add(a, b, scale) = a .+ b .* scale
recursive_add(::Nothing, ::Nothing, scale) = nothing
function recursive_add(a::NT, b::NT, scale = 1) where {NT <: NamedTuple}
    (; (k => recursive_add(a[k], b[k], scale) for k in keys(a))...)
end

function batch_train_step!(states, components, context, train_state, scale = 1; err_fn = MSELoss(), autodiff = AutoZygote())
    (total_grad, init_loss), grads_and_loss = Iterators.peel(Iterators.map(states, components) do state, component
        grads, loss, stats, train_state = Training.compute_gradients(autodiff, err_fn, ((state, context), component), train_state)

        grads, loss
    end)

    loss_stats = StatsAccumulator(init_loss)

    for (grad, loss) in grads_and_loss
        total_grad = recursive_add(total_grad, grad)
        push!(loss_stats, loss)
    end

    loss_stats, Training.apply_gradients!(train_state, total_grad)
end

function learn_components(internal_dims, fock_space::Phi4Space, eigenspace::EigenSpace, energy_levels; activation=tanh)
    model = state_eating_net(subspace_context_dims(fock_space), 1, internal_dims...; activation)

    subhams = sub_hamiltonians(fock_space, eigenspace, energy_levels...)

    rng = Random.default_rng()
    Random.seed!(rng, 0)
    device = gpu_device()

    optimiser = Adam(0.001f0)

    weights, lux_state = Lux.setup(rng, model) |> device
    train_state = Lux.Training.TrainState(model, weights, lux_state, optimiser)

    plt = plot(0)
    ylabel!(plt, "log(loss)")
    xlabel!(plt, "#Datapoints Learned")

    data = Measurement{Float64}[]

    for (E_max, (states, H)) in zip(energy_levels, subhams)
        ground_state, vacuum_energy = groundstate(H)
        
        for i in 1:length(states)
            loss_stats, train_state = batch_train_step!(states, ground_state, context_vec(fock_space, E_max), train_state)

            push!(data, log(total(loss_stats)))
            plot(plt, data)
        end
    end    
end