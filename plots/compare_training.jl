using MLTruncate

output_dims = 1

dim_group_id(dims) = join(dims, "x")
alldims_id(internal_dims) = join(Iterators.map(dim_group_id, internal_dims), "-")
models_id(model_dims) = join(Iterators.map(alldims_id, model_dims), "_")

comma_sep(dims) = isempty(dims) ? "No" : join(dims, ",")

function compare_training!(space, eigenspace, max_energy, coupling, n_epochs, model_dims)
    train_states = map(model_dims) do internal_dims
        setup_model(state_eating_net(output_dims, internal_dims...))
    end

    model_labels = map(model_dims) do (recurrent_dims, processing_dims)
        "$(comma_sep(recurrent_dims)) Reccurent; $(comma_sep(processing_dims)) Processing"
    end

    plot_kwargs = (;
        xlabel="#Training Steps", ylabel=L"\log(MSE)",
        legendtitle="Network Neuron Architectures", legend=:topright, 
        label=permutedims(model_labels)
    )

    plot_name = "plot_training/g=$(coupling)_L=$(size(space))_E=$(max_energy)_nets=$(models_id(model_dims))"

    (; loss_history) = std_cache(plot_name) do
        learn_components!(train_states, space, eigenspace, max_energy, coupling, output_dims, n_epochs;
        plot_kwargs)
    end

    std_savefig(plot(loss_history; plot_kwargs...), plot_name)
end

model_dims = [
    ((10,), (20, 20))
]
max_energy = 10
coupling = 0:0.5:5
n_epochs = 200

compare_training!(FockSpace{Float32}(8), DEFAULT_EIGENSPACE, max_energy, coupling, n_epochs, model_dims)