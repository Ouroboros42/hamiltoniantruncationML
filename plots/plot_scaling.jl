using MLTruncate

function plot_scaling(space, subspace, max_energies)
    plot_name = "scaling/N-$(subspace.n_parity)_E=$(max_energies)"

    sizes = std_cache(plot_name) do 
        map(length, sub_spaces(space, subspace, max_energies))
    end

    title = latexstring("Dimension Scaling of Truncated Fock Space\n\$mL=$(size(space))\$")

    plt = plot(max_energies, sizes;
        xlabel = L"\tilde E_{\mathrm{max}}", ylabel = "Number of States (log-scale)", legend=false,
        yaxis = :log, yticks = (10 .^ (0:5)),
        guidefont = font("Computer Modern", 12)
    )

    std_savefig(plt, plot_name)
end

plot_scaling(FockSpace{Float32}(8), EigenSpace{Int8, UInt8}(x_symmetrisation=Even), 0.1:2:32.1)