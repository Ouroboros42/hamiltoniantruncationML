using MLTruncate

@testset "Known Sequences" begin
    @test sequence_ranks(1:10) == 1:10

    @test sequence_ranks([2, 1, 3.1]) == [2, 1, 3]
end