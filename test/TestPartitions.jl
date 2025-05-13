using MLTruncate

@testset "Known Partitions" begin
    @test collect(energy_ordered_partitions(0)) == [[]]

    @test collect(energy_ordered_partitions(1)) == [[1]]

    @test collect(energy_ordered_partitions(2)) == [[2], [1, 1]]

    @test collect(energy_ordered_partitions(3)) == [[3], [2, 1], [1, 1, 1]]

    @test collect(energy_ordered_partitions(4)) == [[4], [2, 2], [3, 1], [2, 1, 1], [1, 1, 1, 1]]

    @test collect(energy_ordered_partitions(5)) == [[5], [3, 2], [4, 1], [2, 2, 1], [3, 1, 1], [2, 1, 1, 1], [1, 1, 1, 1, 1]]
end