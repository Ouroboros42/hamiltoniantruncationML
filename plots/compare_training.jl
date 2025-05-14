using MLTruncate

output_dims = 1

dim_group_id(dims) = join(dims, "x")
alldims_id(internal_dims) = join(Iterators.map(dim_group_id, internal_dims), "-")
models_id(model_dims) = join(Iterators.map(alldims_id, model_dims), "_")

function compare_training!(space, eigenspace, max_energy, coupling, n_epochs, model_dims, evaluation_)
    train_states = map(model_dims) do internal_dims
        setup_model(state_eating_net(output_dims, internal_dims...))
    end

    model_labels = map(string, model_dims)

    plot_name = sanitise("plot_training/g=$(coupling)_L=$(size(space))_E=$(max_energy)_nets=$(models_id(model_dims))")
    
    (; loss_history, plot_kwargs ) = std_cache(plot_name) do
        learn_components!(train_states, space, eigenspace, max_energy, coupling, output_dims, n_epochs; model_labels)
    end

    plt = plot(loss_history; plot_kwargs..., ylabel=L"\log(MSE)")
    
    std_savefig(plt, plot_name)
end

model_dims = [
    ((10,), (20, 20))
    ((10,), (10, 10))
    ((10,), (20,))
    ((10,), (10,))
]
max_energy = 18
coupling = 2
n_epochs = 100

compare_training!(FockSpace{Float32}(8), DEFAULT_EIGENSPACE, max_energy, coupling, n_epochs, model_dims)