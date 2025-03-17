using Test

@testset "Tests" begin
    @testset "IntPartitions tests" begin
        include("./TestPartitions.jl")
    end

    @testset "State implementation tests" begin
        include("./TestStates.jl")
    end

    @testset "State generation tests" begin
        include("./TestStateGen.jl")
    end
end