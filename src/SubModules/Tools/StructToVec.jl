
"""
    struc_to_vec(s::S) -> Vector{Any}

Return a `Vector{Any}` with one entry per field of `s`. Works when fields
have different concrete types (vectors, matrices, scalars). Efficiently
preallocates the result and preserves each field's value.
"""
function struc_to_vec(s::S) where {S}
    fns = fieldnames(S)
    out = Vector{Any}(undef, length(fns))
    for (i, f) in enumerate(fns)
        out[i] = getfield(s, f)
    end
    return out
end

# Example:
# struct B; a::Vector{Float64}; b::Matrix{Float64}; end
# b = B([1.0,2.0], [3.0 4.0; 5.0 6.0]);
# struc_to_vec(b)  # => 2-element Vector{Any}: [[1.0, 2.0], [3.0 4.0; 5.0 6.0]]

"""
    struc_to_vec_same_type(s::S) -> Vector{T}

Return a `Vector{T}` with one entry per field of `s`, where `T` is the concrete type
of the first field. Asserts that all fields have the same type.
"""
function struc_to_vec_same_type(s::S) where {S}
    fns = fieldnames(S)
    n = length(fns)
    n == 0 && return Vector{Any}(0)
    T = typeof(getfield(s, fns[1]))
    out = Vector{T}(undef, n)
    for (i, f) in enumerate(fns)
        v = getfield(s, f)
        @assert typeof(v) === T "field $f has different type"
        out[i] = v
    end
    return out
end

# Example:
# struct A; a::Vector{Float64}; b::Vector{Float64}; end
# a = A([1.0,2.0], [3.0]); struc_to_vec_same_type(a)  # => Vector{Vector{Float64}} [[1.0,2.0], [3.0]]

"""
    vec_to_struct(::Type{S}, vals::AbstractVector) -> S

Inverse of `struc_to_vec` / `struc_to_vec_same_type`.
Construct an instance of `S` by splatting the elements of `vals` into `S`'s constructor.
Asserts that the number of entries in `vals` matches the number of fields of `S`.
"""
function vec_to_struct(::Type{S}, vals::AbstractVector) where {S}
    @assert length(vals) == length(fieldnames(S)) "length(vals) != number of fields of $S"
    return S(vals...)
end

# # Example usage of functions in StructToVec.jl

# # 1) Different-field types
# struct B
#     a::Vector{Float64}
#     b::Matrix{Float64}
# end

# b = B([1.0, 2.0, 3.0], [3.0 4.0; 5.0 6.0])

# vals = struc_to_vec(b)            # Vector{Any}: [[1.0,2.0,3.0], [3.0 4.0; 5.0 6.0]]
# println(vals)

# # reconstruct B from vals
# b2 = vec_to_struct(typeof(b), vals)
# @assert b2.a == b.a && b2.b == b.b

# # 2) All fields same type -> use struc_to_vec_same_type
# struct A
#     x::Vector{Float64}
#     y::Vector{Float64}
# end

# a = A([1.0,2.0], [3.0])
# vals_same = struc_to_vec_same_type(a)   # Vector{Vector{Float64}}: [[1.0,2.0], [3.0]]
# println(vals_same)

# # reconstruct A
# a2 = vec_to_struct(typeof(a), vals_same)
# @assert a2.x == a.x && a2.y == a.y
