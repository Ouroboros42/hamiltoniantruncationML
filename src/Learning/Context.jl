using ..Hamiltonian

export context_dims, subspace_context_dims, context_vec

context_dims(s) = length(context_vec(s))
subspace_context_dims(s) = context_dims(s) + 1

context_vec(x::Real) = [ x ]
context_vec(v::Vector{R}) where {R <: Real} = v
context_vec(space::BoundedFockSpace) = [ k_unit(space) ]
context_vec(space::Phi4Space) = [ k_unit(space), coupling(space) ]

context_vec(args...) = vcat(map(context_vec, args)...)