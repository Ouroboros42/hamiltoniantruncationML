export PLOT_ROOT, PLOT_OUT, PLOT_CACHE, sanitise, std_cache, std_savefig, freeze_lims!, freeze_xlims!, freeze_ylims!

using Reexport
@reexport using Plots
@reexport using StatsPlots
@reexport using CacheVariables
@reexport using LaTeXStrings
@reexport using LogExpFunctions

function __init__()
    gr(show = true)

    default(;
        guidefont = font("Computer Modern", 11),
        titlefont = font("Computer Modern", 11),
        legendfont = font("Computer Modern", 8),
        legendtitlefontfamily = "Computer Modern",
        legendtitlefontsize = 9,
        palette = :Set1_9
    )
end

const PLOT_ROOT = "$(dirname(@__DIR__))/plots"
const PLOT_CACHE = "$PLOT_ROOT/cache"
const PLOT_OUT = "$PLOT_ROOT/output"

function sanitise(path)
    replace(path, ":" => "꞉")
end

function std_cache(fn, plot_name)
    output_path = "$PLOT_CACHE/$(sanitise(plot_name)).bson"
    
    cache(fn, output_path)
end

function std_savefig(fig, plot_name)
    output_path = "$PLOT_OUT/$(sanitise(plot_name)).pdf"

    mkpath(dirname(output_path))
    savefig(fig, output_path)

    @info "Saving fig to $output_path"
end

function freeze_ylims!(plt)
    plot!(plt, ylim=ylims(plt))
end

function freeze_xlims!(plt)
    plot!(plt, xlim=xlims(plt))
end

function freeze_lims!(plt)
    freeze_xlims!(plt)
    freeze_ylims!(plt)
end