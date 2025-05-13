using MLTruncate

S = DictFockState{Int8, UInt8}

states = [
    S(0 => 2)
    S(1 => 1, 2 => 1)
    S(3 => 1, 0 => 1)
]

space = FockSpace(10)

Mint = compute(Phi4Interaction(space), states)

diag_predicted(k1, k2) = 1 / (free_energy(space, k1) * free_energy(space, k2)) 
offdiag_predicted = 1 / sqrt(free_energy(space, 0) * free_energy(space, 1) * free_energy(space, 2) * free_energy(space, 3))

@test Mint[2, 2] / Mint[3, 3] ≈ diag_predicted(1, 2) / diag_predicted(0, 3)
@test Mint[2, 3] / Mint[2, 2] ≈ offdiag_predicted / diag_predicted(1, 2)
