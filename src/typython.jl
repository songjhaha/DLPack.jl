# SPDX-License-Identifier: MIT
# See LICENSE.md at https://github.com/pabloferz/DLPack.jl


const TyCPython = TyPython.CPython

# This will be used to release the `DLManagedTensor`s and associated array.
const TYPYTHON_DLPACK_DELETER = @cfunction(release, Cvoid, (Ptr{Cvoid},))

const PYCAPSULE_NAME_TYPYTHON = "dltensor"
const USED_PYCAPSULE_NAME_TYPYTHON = "used_dltensor"

"""
    DLManagedTensor(po::Py)

Takes a PyCapsule holding a `DLManagedTensor` and returns the latter.
"""
function DLManagedTensor(po::TyCPython.Py)
    ptr = TyCPython.unsafe_unwrap(po)

    if TyCPython.PyAPI.PyObject_IsInstance(ptr, TyCPython.PyAPI.PyCapsule_Type) != 1
        throw(ArgumentError("PyObject must be a PyCapsule"))
    end

    name = TyCPython.PyAPI.PyCapsule_GetName(ptr)

    if unsafe_string(name) == USED_PYCAPSULE_NAME_TYPYTHON
        throw(ArgumentError("PyCapsule in use, have you wrapped it already?"))
    end

    dlptr = Ptr{DLManagedTensor}(TyCPython.PyAPI.PyCapsule_GetPointer(ptr, name))
    tensor = DLManagedTensor(dlptr)

    # Replace the capsule name to "used_dltensor"
    set_name_flag = TyCPython.PyAPI.PyCapsule_SetName(ptr, USED_PYCAPSULE_NAME_TYPYTHON)

    if set_name_flag != 0
        @warn("Could not mark PyCapsule as used")
        TyCPython.py_throw()
    end

    # Extra precaution: Replace the capsule destructor to prevent it from deleting the
    # tensor. We will use the `DLManagedTensor.deleter` instead.
    if TyCPython.PyAPI.PyCapsule_SetDestructor(ptr, C_NULL) != 0
        @warn("Could not replace the capsule destructor")
        TyCPython.py_throw()
    end

    return tensor
end

"""
    wrap(o::Py, to_dlpack)

Takes a tensor `o::Py` and a `to_dlpack` function that generates a
`DLManagedTensor` bundled in a PyCapsule, and returns a zero-copy
`array::AbstractArray` pointing to the same data in `o`.
For tensors with row-major ordering the resulting array will have all
dimensions reversed.
"""
function wrap(o::TyCPython.Py, to_dlpack::Union{TyCPython.Py, Function})
    return unsafe_wrap(DLManagedTensor(to_dlpack(o)), o)
end

"""
    wrap(::Type{<: AbstractArray{T, N}}, ::Type{<: MemoryLayout}, o::Py, to_dlpack)

Type-inferrable alternative to `wrap(o, to_dlpack)`.
"""
function wrap(::Type{A}, ::Type{M}, o::TyCPython.Py, to_dlpack) where {
    T, N, A <: AbstractArray{T, N}, M
}
    return unsafe_wrap(A, M, DLManagedTensor(to_dlpack(o)), o)
end

"""
    share(A::StridedArray, from_dlpack::Py)

Takes a Julia array and an external `from_dlpack` method that consumes PyCapsules
following the DLPack protocol. Returns a Python tensor that shares the data with `A`.
The resulting tensor will have all dimensions reversed with respect
to the Julia array.
"""
share(A::StridedArray, from_dlpack::TyCPython.Py) = share(A, TyCPython.Py, from_dlpack)

"""
    share(A::StridedArray, ::Type{Py}, from_dlpack)

Similar to `share(A, from_dlpack::Py)`. Use when there is a need to
disambiguate the return type.
"""
function share(A::StridedArray, ::Type{TyCPython.Py}, from_dlpack)
    capsule = share(A)
    tensor = capsule.tensor
    tensor_ptr = pointer_from_objref(tensor)

    # Prevent `A` and `tensor` from being `gc`ed while `o` is around.
    # For certain DLPack-compatible libraries, e.g. PyTorch, the tensor is
    # captured and the `deleter` referenced from it.
    SHARES_POOL[tensor_ptr] = (capsule, A)
    tensor.deleter = TYPYTHON_DLPACK_DELETER

    pycapsule = TyCPython.Py(
        TyCPython.PyAPI.PyCapsule_New(tensor_ptr, PYCAPSULE_NAME_TYPYTHON, C_NULL)
    )

    return from_dlpack(pycapsule)
end
