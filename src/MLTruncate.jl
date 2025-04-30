module MLTruncate

module Hamiltonian

include("Hamiltonian/Util/Conversions.jl")
include("Hamiltonian/Util/Parity.jl")
include("Hamiltonian/Util/IntPartitions.jl")
include("Hamiltonian/Util/Stats.jl")
include("Hamiltonian/FockSpace/States.jl")
include("Hamiltonian/FockSpace/Symmetrised.jl")
include("Hamiltonian/FockSpace/FockSpace.jl")
include("Hamiltonian/FockSpace/BuildMatrix.jl")
include("Hamiltonian/FockSpace/SubSpace.jl")
include("Hamiltonian/FockSpace/StateGen.jl")
include("Hamiltonian/FockSpace/Diagonalisation.jl")
include("Hamiltonian/FockSpace/Stratified.jl")
include("Hamiltonian/Interaction/Ladders.jl")
include("Hamiltonian/Interaction/Phi4.jl")

end

module Learning

include("Learning/Context.jl")
include("Learning/NetworkStructure.jl")
include("Learning/Training.jl")

end

end