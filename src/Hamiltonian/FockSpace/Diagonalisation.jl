using ArnoldiMethod

export spectrum, groundstate

function spectrum(hamiltonian, n_eigs::Integer)
    decomp, history = partialschur(hamiltonian; nev=n_eigs, which=:SR)

    println("Diagonalisation: $history")

    eigvecs = decomp.Q
    real_eigvals = Real.(decomp.eigenvalues)
    
    eigvecs[:, begin:begin+n_eigs-1], real_eigvals[begin:begin+n_eigs-1]
end

spectrum(hamiltonian) = spectrum(hamiltonian, size(hamiltonian, 1))

function groundstate(hamiltonian)
    eigvecs, eigvals = spectrum(hamiltonian, 1)
    
    eigvecs[:, begin], eigvals[begin]
end