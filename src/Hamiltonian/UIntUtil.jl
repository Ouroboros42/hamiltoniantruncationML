function force_unsigned(n::Signed)
    if n < 0
        throw("Positive integer expected, got $n")
    end

    unsigned(n)
end
