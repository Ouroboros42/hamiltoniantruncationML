using ArnoldiMethod

export spectrum

function spectrum(hamiltonian, n_eigs::Integer)
    decomp, history = partialschur(hamiltonian; nev=n_eigs, which=:SR)

    println("Diagonalisation: $history")

    eigvecs = decomp.Q
    real_eigvals = Real.(decomp.eigenvalues)
    
    eigvecs, real_eigvals
end

spectrum(hamiltonian) = spectrum(hamiltonian, size(hamiltonian, 1))
