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

function sort_scored(scorer, states)
    scored = map(states) do state
        ScoredState(state, scorer(state))
    end

    sort!(scored, by=getscores, rev=true)
end

sort_scored(scorer, states, n_selected) = first(sort_scored(scorer, states), n_selected)
sort_by_score(args...) = map(getstates, sort_scored(args...))
