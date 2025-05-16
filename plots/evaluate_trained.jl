using MLTruncate

function evaluate_trained(model, space, eigenspace, low_max_energy, couplings, n_epochs, high_max_energy, n_extra_states)
    plot_kwargs = (;
        xlabel="#Training Epochs", ylabel=L"\log\left(\overline{\mathrm{Err}^2}\right)",
        legend=:topright, 
        label=[ "Uniform baseline" "Trained Network" ],
        linealpha=0.6
    )

    context_kwargs = (; cutoff_in_context = false)

    file_name(section) = "evaluate_trained/$section/N-$(eigenspace.n_parity)_epochs=$(n_epochs)_g=$(couplings)_L=$(size(space))_E=$(low_max_energy)-$(high_max_energy)_n+$(n_extra_states)"

    (; parameters, loss_history, subspaces) = std_cache(file_name("training")) do
        learn_components!([setup_model(model, 0.005)], space, eigenspace, max_energy, couplings, output_dims, n_epochs;
            plot_kwargs, baseline_predictors = [uniform_baseline], context_kwargs...
        )
    end

    trained_model = restart_model(model, only(parameters))

    std_savefig(plot(loss_history; plot_kwargs...), file_name("training"))

    (; baseline_E0s, trained_E0s) = std_cache(file_name("applied")) do
        shared_states = first(subspaces).states

        more_states = collect(Iterators.filter(!in(shared_states), generate_states(space, eigenspace, high_max_energy)))
        max_extra_states = n_extra_states[end]
        baseline_scorer = -free_energy(space)

        baseline_subspaces = scored_subspaces(baseline_scorer, more_states, n_extra_states, shared_states)
        baseline_subhams = any_sub_hamiltonians(space, baseline_subspaces, nothing, permutedims(couplings))

        baseline_E0s = map(baseline_subhams) do (; hamiltonian)
            groundenergy(hamiltonian)
        end

        trained_E0s = mapreduce(hcat, subspaces) do (; coupling, max_energy)
            context = make_context(space, coupling, max_energy; context_kwargs...)

            trained_scorer = state_scorer(trained_model, context)
            trained_subspaces = scored_subspaces(trained_scorer, more_states, n_extra_states, shared_states)
            map(any_sub_hamiltonians(space, trained_subspaces, nothing, coupling)) do (; hamiltonian)
                groundenergy(hamiltonian)[1]
            end
        end

        (; baseline_E0s, trained_E0s)
    end

    plot(n_extra_states, [baseline_E0s[:, 3] trained_E0s[:, 3]]; label = [ "Free Energy" "Trained Model" ], legendtitle = "Ordering")
end

model = state_eating_net(1, (20,), (30,), context_dims=2)
space = FockSpace{Float32}(8)

low_max_energy = 10
high_max_energy = 20
couplings = 1:0.5:5
n_epochs = 1000
n_extra_states = 0:500:3000

for n_parity in (Even, Odd)
    eigenspace = FastEigenSpace(; n_parity)

    evaluate_trained(model, space, eigenspace, low_max_energy, couplings, n_epochs, high_max_energy, n_extra_states)
end