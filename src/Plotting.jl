export PLOT_ROOT, PLOT_OUT, PLOT_CACHE, sanitise

using Reexport
@reexport using Plots
@reexport using CacheVariables

function __init__()
    gr(show = true)
end

const PLOT_ROOT = "$(dirname(@__DIR__))/plots"
const PLOT_CACHE = "$PLOT_ROOT/cache"
const PLOT_OUT = "$PLOT_ROOT/output"

function sanitise(path)
    replace(path, ":" => "꞉")
end