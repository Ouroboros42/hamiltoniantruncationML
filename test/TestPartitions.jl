using MLTruncate.Hamiltonian.IntPartitions

@testset "Known Partitions" begin
    @test collect(PartitionBuilder(0)) == [[]]

    @test collect(PartitionBuilder(1)) == [[1]]

    @test collect(PartitionBuilder(2)) == [[2], [1, 1]]

    @test collect(PartitionBuilder(3)) == [[3], [2, 1], [1, 1, 1]]

    @test collect(PartitionBuilder(4)) == [[4], [2, 2], [3, 1], [2, 1, 1], [1, 1, 1, 1]]

    @test collect(PartitionBuilder(5)) == [[5], [3, 2], [4, 1], [2, 2, 1], [3, 1, 1], [2, 1, 1, 1], [1, 1, 1, 1, 1]]
end