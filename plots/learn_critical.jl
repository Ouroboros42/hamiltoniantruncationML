using MLTruncate

function evaluate_mphys(space, low_Emax, high_Emax, n_selected_states, couplings, n_epochs, recurrent_dims, processing_dims, K = Int8, N = UInt8)
    E0, E1 = map((Even, Odd)) do Pn
        eigenspace = EigenSpace{K, N}(x_symmetrisation=Even, n_parity = Pn)

        model = setup_model(state_eating_net(1, recurrent_dims, processing_dims))
        learn_components!(model, space, eigenspace, low_Emax, couplings, 1, n_epochs)

        map(couplings) do coupling
            scored_states = map(state_scorer(model, space, coupling, low_Emax), generate_states(space, eigenspace, high_Emax))
            sort!(scored_states, by=getscores, rev=true)
            selected_states = map(getstates, scored_states[1:min(n_selected_states, end)])

            H = compute(hamiltonian(space, coupling), selected_states)

            groundstate(H)[2]
        end
    end

    mphys = E1 .- E0
    
    plot(couplings, mphys, xlabel = "g", ylabel = "Mphys")
end

low_Emax = 18
n_selected_states = 3000
high_Emax = 26
couplings = 0:0.5:5
n_epochs = 2
recurrent_dims = (40,)
processing_dims = (40, 40)
evaluate_mphys(FockSpace{Float32}(8), low_Emax, high_Emax, n_selected_states, couplings, n_epochs, recurrent_dims, processing_dims)