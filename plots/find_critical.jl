using MLTruncate.Hamiltonian
using Plots

function evaluate_mphys(free_space, max_energy, couplings, K = Int8, N = UInt8)
    (E0, E1) = map((Even, Odd)) do Pn
        eigenspace = EigenSpace{K, N}(x_symmetrisation=Even, n_parity = Pn)

        subhams = sub_hamiltonians(free_space, eigenspace, (max_energy,), couplings)

        map(subhams) do (states, H)
            _, E = groundstate(H)
            E
        end
    end

    mphys = E1 .- E0
    
    plot(couplings, mphys, xlabel = "g", ylabel = "Mphys")
end

evaluate_mphys(FockSpaceImpl(Float32(2π / 8)), 17, 0:.01:5)