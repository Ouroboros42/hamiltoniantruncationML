function evaluate_trained(model, space, eigenspace, low_max_energy, couplings, n_epochs, high_max_energy, n_extra_states)
    plot_kwargs = (;
        xlabel="#Training Epochs", ylabel=L"\log\left(\overline{\mathrm{Err}^2}\right)",
        legend=:topright, 
        label=[ "Uniform baseline" "Trained Network" ],
        linealpha=0.6
    )

    file_name(section) = "evaluate_trained/$section/N-$(eigenspace.n_parity)_epochs=$(n_epochs)_g=$(couplings)_L=$(size(space))_E=$(low_max_energy)-$(high_max_energy)_n+$(n_extra_states)"

    (; train_states, loss_history, subspaces) = std_cache(file_name("training")) do
        learn_components!([setup_model(model, 0.005)], space, eigenspace, max_energy, couplings, output_dims, n_epochs;
            plot_kwargs, baseline_predictors = [uniform_baseline],
            cutoff_in_context = false
        )
    end

    std_savefig(plot(loss_history; plot_kwargs...), file_name("training"))
end

model = state_eating_net(1, (20,), (30,), context_dims=2)
space = FockSpace{Float32}(8)

low_max_energy = 10
high_max_energy = 20
couplings = 1:0.5:5
n_epochs = 1000
n_extra_states = 0:500:3000

for n_parity in (Even,)
    eigenspace = FastEigenSpace(; n_parity)

    evaluate_trained(model, space, eigenspace, low_max_energy, couplings, n_epochs, high_max_energy, n_extra_states)
end