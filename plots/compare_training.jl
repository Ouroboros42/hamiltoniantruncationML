using MLTruncate

using LogExpFunctions: softplus

output_dims = 1

dim_group_id(dims) = join(dims, "x")
alldims_id(internal_dims) = join(Iterators.map(dim_group_id, internal_dims), "-")
models_id(model_dims) = join(Iterators.map(alldims_id, model_dims), "_")

comma_sep(dims) = isempty(dims) ? "No" : join(dims, ",")

uniform_baseline(true_components) = fill(sum(true_components) / length(true_components), length(true_components))

function compare_training!(space, eigenspace, max_energy, coupling, n_epochs, model_dims, model_acts)
    baseline_labels = [
        "Uniform baseline"
    ]

    model_labels = map(model_dims, model_acts) do (recurrent_dims, processing_dims), (internal_act, out_act)
        "Neurons: $(comma_sep(recurrent_dims)) Reccurent; $(comma_sep(processing_dims)) Processing, Activation = $internal_act, $out_act"
    end

    plot_kwargs = (;
        xlabel="#Training Epochs", ylabel=L"\log\left(\overline{\mathrm{Err}^2}\right)",
        legend=:topright, 
        label=permutedims(vcat(baseline_labels, model_labels)),
        linealpha=0.6
    )

    plot_name = "plot_training/epochs=$(n_epochs)_g=$(coupling)_L=$(size(space))_E=$(max_energy)_nets=$(models_id(model_dims))"

    (; train_states, loss_history) = std_cache(plot_name) do
        train_states = map(model_dims, model_acts) do internal_dims, (activation, out_activation)
            setup_model(state_eating_net(output_dims, internal_dims...; activation, out_activation))
        end

        learn_components!(train_states, space, eigenspace, max_energy, coupling, output_dims, n_epochs;
            plot_kwargs, baseline_predictors = [uniform_baseline]
        )
    end

    std_savefig(plot(loss_history; plot_kwargs...), plot_name)
end

model_dims = [
    ((10,), (30,))
    ((10,), (30,))
]

model_acts = [
    (tanh, logistic)
    (softplus, logistic)
]

max_energy = 15
coupling = 1:0.5:5
n_epochs = 500

compare_training!(FockSpace{Float32}(8), DEFAULT_EIGENSPACE, max_energy, coupling, n_epochs, model_dims, model_acts)