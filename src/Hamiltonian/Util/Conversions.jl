force_unsigned(n::Unsigned) = n

function force_unsigned(n::Signed)
    if n < 0
        throw("Positive integer expected, got $n")
    end

    unsigned(n)
end

promoteto(target, values::R...) where R = map(target, values)
promoteto(target, values...) = promoteto(target, promote(values...)...)