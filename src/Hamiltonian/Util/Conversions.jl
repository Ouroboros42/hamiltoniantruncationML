force_unsigned(n::Unsigned) = n

function force_unsigned(n::Signed)
    if n < 0
        throw("Positive integer expected, got $n")
    end

    unsigned(n)
end

promoteto(target, values::R...) where R = map(target, values)
promoteto(target, values...) = promoteto(target, promote(values...)...)

function sign_split(::Type{Nout}, plus::Nin, minus::Nin = zero(Nin))::Tuple{Nout, Nout} where {Nin, Nout}
    plus >= minus ? (plus - minus, 0) : (0, minus - plus)
end

sign_split(plus::N, minus = zero(N)) where N = sign_split(unsigned(N), plus, minus)