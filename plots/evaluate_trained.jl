using MLTruncate
using Statistics

function evaluate_trained(training_file_name, crit_file_name, model, space, eigenspace, low_max_energy, couplings, n_epochs, high_max_energy, n_extra_states, alpha, use_title::Bool)
    ptitle = use_title ? "$(eigenspace.n_parity) Subspace" : ""

    plot_kwargs = (;
        xlabel="Training Epochs", ylabel=L"\log\left(\mathrm{MSE}\right)",
        legend=:topright, 
        title = ptitle,
        label=[ "Uniform baseline" "Neural Network" ]
    )

    context_kwargs = (; cutoff_in_context = false)

    (; parameters, loss_history, subspaces, solutions) = std_cache(training_file_name("training")) do
        learn_components!([setup_model(model, alpha)], space, eigenspace, low_max_energy, couplings, 1, n_epochs;
            plot_kwargs, baseline_predictors = [uniform_baseline], context_kwargs...
        )
    end

    parameters = only(parameters)

    trained_model = restart_model(model, parameters)

    std_savefig(plot(loss_history; plot_kwargs...), training_file_name("training"))

    (all_actual_components, all_predicted_components) = std_cache(training_file_name("sample-all")) do
        mapreduce((a, b) -> vcat.(a, b), subspaces, solutions) do (; states), (; context, components)
            (
                components[2:end],
                apply(trained_model, context, states)[2:end, 1, 1]
            )
        end
    end

    begin
        (; weight, bias) = first(parameters).last_layer

        lower_bound = exp(only(bias) - sum(abs, weight))

        comp_plt = scatter(all_actual_components, all_predicted_components; label = "Trained Model",
            xaxis = :log, yaxis = :log,
            xlabel = "Actual Ground State Component (log-scale)",
            ylabel = "Model Prediction (log-scale)",
            markersize = 1.5,
            markerstrokewidth = 0,
            markercolor = :black,
            title = ptitle,
            legend = :topleft
        )

        freeze_xlims!(comp_plt)

        coord_range = [eps(), 1]

        # plot!(comp_plt, coord_range, [lower_bound, lower_bound]; label = "Model Bound")

        freeze_ylims!(comp_plt)

        plot!(comp_plt, coord_range, coord_range; linewidth = 2, label = "Objective", linealpha = 0.8)

        std_savefig(comp_plt, training_file_name("sample-all"))
    end

    shared_states = first(subspaces).states

    (; baseline_E0s, trained_grounds) = std_cache(crit_file_name("applied")) do
        more_states = collect(Iterators.filter(!in(shared_states), generate_states(space, eigenspace, high_max_energy)))
        baseline_scorer = -free_energy(space)

        baseline_subspaces = scored_subspaces(baseline_scorer, more_states, n_extra_states, shared_states)
        baseline_subhams = any_sub_hamiltonians(space, baseline_subspaces, nothing, permutedims(couplings))

        baseline_E0s = map(baseline_subhams) do (; hamiltonian)
            groundenergy(hamiltonian)
        end

        trained_grounds = mapreduce(hcat, subspaces) do (; coupling, max_energy)
            context = make_context(space, coupling, max_energy; context_kwargs...)

            trained_scorer = state_scorer(trained_model, context)
            trained_subspaces = scored_subspaces(trained_scorer, more_states, n_extra_states, shared_states)
            map(any_sub_hamiltonians(space, trained_subspaces, nothing, coupling)) do (; hamiltonian, states)
                (trained_scorer(states), groundstate(hamiltonian)...)
            end
        end

        (; baseline_E0s, trained_grounds)
    end

    n_states = n_extra_states .+ length(shared_states)

    begin
        correlation = map(trained_grounds) do (scores, components, _)
            scores = dropdims(scores; dims=(2, 3))
            cor(scores, abs.(components))
        end

        corrplt = plot(n_states, correlation;
            legendtitle = L"\tilde g",
            label = permutedims(couplings),
            palette = :managua10,
            ylim = [0, 1],
            xlabel = "Truncated Subspace Dimension",
            ylabel = "Pearson Correlation Coefficient",
        )

        std_savefig(corrplt, crit_file_name("correlation"))
        
        scores = map(g -> dropdims(g[1], dims=(2, 3)), trained_grounds)
        trained_components = map(g -> g[2], trained_grounds)

        ig = 1
        iN = 1

        select_comps(m) = m[iN, ig]

        selected_scores = select_comps(scores)
        selected_components = abs.(select_comps(trained_components))

        compplt = scatter(selected_scores, selected_components;
            xaxis = :log, yaxis = :log,
            title = ptitle,

        )

        std_savefig(compplt, crit_file_name("components"))
    end

    trained_E0s = map(g -> g[3], trained_grounds)

    E0_diff = @. (trained_E0s - baseline_E0s)

    plt = plot(n_states, E0_diff;
        title = ptitle,
        legendtitle = L"\tilde g",
        label = permutedims(couplings),
        xlabel = "Truncated Subspace Dimension",
        ylabel = L"\tilde \mathcal{E}_\mathrm{NN} - \tilde \mathcal{E}_\mathrm{FE}",
        palette = :managua10
    )

    std_savefig(plt, crit_file_name("applied"))

    baseline_E0s[end, :], trained_E0s[end, :]
end

model = state_eating_net(1, (10,), (20,), context_dims=2)
space = FockSpace{Float32}(8)

display(model)

low_max_energy = 14
high_max_energy = 25
couplings = 0.5:0.5:5
n_epochs = 3000
n_extra_states = 0:1000:6000
alpha = 0.001
sample_i = 4

training_file_name(n_parity) =  section -> "evaluate_trained/$section/N-$(n_parity)_epochs=$(n_epochs)_g=$(couplings)_L=$(size(space))_E=$(low_max_energy)-$(high_max_energy)"
crit_file_name(n_parity) =  section -> "evaluate_trained/$section/N-$(n_parity)_epochs=$(n_epochs)_g=$(couplings)_L=$(size(space))_E=$(low_max_energy)-$(high_max_energy)_n+$(n_extra_states)"

(bE0, tE0), (bE1, tE1) = map((Even, Odd)) do n_parity
    eigenspace = FastEigenSpace(; n_parity)

    evaluate_trained(training_file_name(n_parity), crit_file_name(n_parity), model, space, eigenspace, low_max_energy, couplings, n_epochs, high_max_energy, n_extra_states, alpha, false)
end

bmphys = bE1 .- bE0
tmphys = tE1 .- tE0

crit_plot_kwargs = (;
    legendtitle = "State Selection",
    label = [ "Free Energy" "Trained Model" ],
    xlabel = L"\tilde g", ylabel = L"\tilde m_{\mathrm{phys}}",
)

plt = plot(couplings, [ bmphys tmphys ]; crit_plot_kwargs...)
std_savefig(plt, crit_file_name("diff")("critplot"))