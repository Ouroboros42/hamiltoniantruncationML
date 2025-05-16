export state_scorer, sort_by_score, scored_subspaces

import Base: -

-(f) = x -> -f(x)

struct ScoredState{S, F}
    state::S
    score::F
end

getstates(scored::ScoredState) = scored.state
getscores(scored::ScoredState) = scored.score

state_scorer(trainstate::TrainState, context) = state -> apply(trainstate, (context, state))
state_scorer(trainstate::TrainState, args...) = state_scorer(trainstate, make_context(args...))
state_scorer(model, context, params) = state -> first(apply(model, (context, [state]), params...))

function sort_scored(scorer, states)
    scored = map(states) do state
        ScoredState(state, scorer(state))
    end

    sort!(scored, by=getscores, rev=true)
end

sort_scored(scorer, states, n_selected) = first(sort_scored(scorer, states), n_selected)
sort_by_score(args...) = map(getstates, sort_scored(args...))

function scored_subspaces(scorer, states, n_selections, extra_states)
    ordered_states = sort_by_score(scorer, states)
    
    map(n_selections) do n
        [ extra_states; ordered_states[1:n] ]
    end
end