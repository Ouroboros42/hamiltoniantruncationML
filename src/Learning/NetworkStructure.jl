export state_eating_net

using ..Hamiltonian
using Lux, LogExpFunctions

function preprocess(state::DictFockState)
    collect(Iterators.map(state) do (k, n)
        [Float32(k), Float32(n)]
    end)
end
preprocess(state::SymmetrisedFockState) = preprocess(state.base_state)

const PREPROCESSOR_LAYER = WrappedFunction(preprocess)

function state_eating_layer(internal_state_dims, activation=tanh)
    Chain(
        PREPROCESSOR_LAYER,
        Recurrence(RNNCell(2 => internal_state_dims, activation, train_state=true))
    )
end

function state_eating_net(context_dims, output_dims, state_encoding_dims, hidden_layer_dims...; activation = tanh, out_activation = logistic)
    state_eater = state_eating_layer(state_encoding_dims, activation)
    state_and_context_eater = PairwiseFusion(vcat, state_eater)
    
    prev_dim = state_encoding_dims + context_dims
    hidden_layers = map(hidden_layer_dims) do next_dim
        layer = Dense(prev_dim, next_dim, activation)
        prev_dim = next_dim
        layer
    end

    output_layer = Dense(hidden_layer_dims[end], output_dims, out_activation)

    Chain(state_and_context_eater, hidden_layers..., output_layer)
end