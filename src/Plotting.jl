export PLOT_ROOT, PLOT_OUT, PLOT_CACHE, sanitise, std_cache, std_savefig

using Reexport
@reexport using Plots
@reexport using StatsPlots
@reexport using CacheVariables
@reexport using LaTeXStrings

function __init__()
    gr(show = true)
end

const PLOT_ROOT = "$(dirname(@__DIR__))/plots"
const PLOT_CACHE = "$PLOT_ROOT/cache"
const PLOT_OUT = "$PLOT_ROOT/output"

function sanitise(path)
    replace(path, ":" => "꞉")
end

function std_cache(fn, plot_name)
    cache(fn, "$PLOT_CACHE/$plot_name.bson")
end

function std_savefig(fig, plot_name)
    output_path = "$PLOT_OUT/$plot_name.pdf"

    mkpath(dirname(output_path))
    savefig(fig, output_path)
end