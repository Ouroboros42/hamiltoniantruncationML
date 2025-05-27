using MLTruncate

using LogExpFunctions: softplus

output_dims = 1

dim_group_id(dims) = join(dims, "x")
alldims_id(internal_dims) = join(Iterators.map(dim_group_id, internal_dims), "-")
models_id(model_dims) = join(Iterators.map(alldims_id, model_dims), "_")

comma_sep(dims) = isempty(dims) ? "No" : join(dims, ",")

function compare_training!(space, eigenspace, max_energy, coupling, n_epochs, model_dims)
    baseline_labels = [
        "Uniform baseline"
    ]

    model_labels = map(model_dims) do (recurrent_dims, processing_dims)
        "Layer sizes: $(comma_sep(recurrent_dims)) Reccurent; $(comma_sep(processing_dims)) Hidden"
    end

    plot_kwargs = (;
        xlabel="#Training Epochs", ylabel=L"\log\left(\mathrm{MSE}\right)",
        legend=:topright,
        label=permutedims(vcat(baseline_labels, model_labels)),
        linealpha=0.6
    )

    plot_name = "plot_training/N-$(eigenspace.n_parity)_epochs=$(n_epochs)_g=$(coupling)_L=$(size(space))_E=$(max_energy)_nets=$(models_id(model_dims))"

    (; loss_history) = std_cache(plot_name) do
        train_states = map(model_dims) do internal_dims
            model = state_eating_net(output_dims, internal_dims...; context_dims=2)

            display(model)

            setup_model(model)
        end

        learn_components!(train_states, space, eigenspace, max_energy, coupling, output_dims, n_epochs;
            plot_kwargs, baseline_predictors = [uniform_baseline], cutoff_in_context = false
        )
    end

    std_savefig(plot(loss_history; plot_kwargs...), plot_name)
end

model_dims = [
    ((10,), (20,))
]

max_energy = 14
coupling = 0.5:0.5:5
n_epochs = 2000

for n_parity in (Even, Odd)
    compare_training!(FockSpace{Float32}(8), EigenSpace{Int8, UInt8}(; n_parity), max_energy, coupling, n_epochs, model_dims)
end