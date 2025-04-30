export state_eating_net

using ..Hamiltonian
using Lux, LogExpFunctions

adjacent_pairs(iter) = zip(iter, Iterators.drop(iter, 1))

function dense_layers(neuron_counts, args...; kwargs...)
    [ Dense(input, output, args...; kwargs...) for (input, output) in adjacent_pairs(neuron_counts)]
end

function apply_layers(layers, input)
    for layer in layers
        input = layer(input)
    end

    input
end

function state_eating_net(context_dims, output_dims, state_layer_dims, hidden_layer_dims; activation = tanh, out_activation = logistic)
    state_encoding_dims = state_layer_dims[end]

    @compact(
        initial_encoding = zeros(state_encoding_dims),
        state_encoding_layers = dense_layers((2 + context_dims + state_encoding_dims, state_layer_dims...), activation),
        processing_layers = dense_layers((state_encoding_dims, hidden_layer_dims...), activation),
        last_layer = Dense(hidden_layer_dims[end], output_dims, out_activation)
    ) do (states, context)
        output = mapreduce(vcat, states) do state
            encoding = initial_encoding

            for (momentum, count) in pairs(representative_fockstate(state))
                encoding = apply_layers(state_encoding_layers, [ momentum; count; context; encoding ])
            end

            permutedims(last_layer(apply_layers(processing_layers, encoding)))
        end

        @return output
    end
end