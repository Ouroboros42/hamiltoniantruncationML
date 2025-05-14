export sequence_ranks

function sequence_ranks(scores)
    indexed_elements = collect(enumerate(scores))
    sort!(indexed_elements, by=x->x[2])
    
    index_pairs = map(enumerate(indexed_elements)) do (i_sorted, (i_original, _))
        (i_sorted, i_original)
    end

    sort!(index_pairs, by=x->x[2])
    map(first, index_pairs)
end
