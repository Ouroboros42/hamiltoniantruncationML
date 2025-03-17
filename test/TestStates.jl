using MLTruncate.Hamiltonian

@testset "Equality tests" begin
    @test DictFockState(1 => 1, 0 => 2) == DictFockState(1 => 1, 0 => 2)
    @test DictFockState(1 => 0, 2 => 1) == DictFockState(2 => 1)
    @test DictFockState(1 => 3) != DictFockState(1 => 2)
end