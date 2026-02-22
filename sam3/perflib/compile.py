# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

import torch


def recursive_fn_factory(fn):
    def recursive_fn(b):
        if isinstance(b, dict):
            return {k: recursive_fn(b[k]) for k in b}
        if isinstance(b, list):
            return [recursive_fn(t) for t in b]
        if isinstance(b, tuple):
            return tuple(recursive_fn(t) for t in b)
        if isinstance(b, torch.Tensor):
            return fn(b)
        # Yes, writing out an explicit white list of
        # trivial types is tedious, but so are bugs that
        # come from not applying fn, when expected to have
        # applied it.
        if b is None:
            return b
        trivial_types = [bool, int]
        for t in trivial_types:
            if isinstance(b, t):
                return b
        raise TypeError(f"Unexpected type {type(b)}")

    return recursive_fn


recursive_contiguous = recursive_fn_factory(lambda x: x.contiguous())
recursive_clone = recursive_fn_factory(torch.clone)


def compile_wrapper(
    fn,
    *,
    mode="max-autotune",
    fullgraph=True,
    dynamic=False,
    name=None,
    make_contiguous=True,
    clone_output=True,
):
    compiled_fn = torch.compile(fn, mode=mode, fullgraph=fullgraph, dynamic=dynamic)
    # Avoid runtime `repr(fn)` in every call; module/method repr can be expensive.
    if name is not None:
        profile_name = name
    else:
        fn_name = getattr(fn, "__qualname__", None) or getattr(fn, "__name__", None)
        profile_name = f"compiled {fn_name}" if fn_name is not None else "compiled_fn"


    def compiled_fn_wrapper(*args, **kwargs):
        with torch.autograd.profiler.record_function(
            profile_name
        ):
            if make_contiguous:
                cont_args = recursive_contiguous(args)
                cont_kwargs = recursive_contiguous(kwargs)
            else:
                cont_args = args
                cont_kwargs = kwargs
            result = compiled_fn(*cont_args, **cont_kwargs)
            if clone_output:
                return recursive_clone(result)
            return result

    return compiled_fn_wrapper


def shape_logging_wrapper(fn, keep_kwargs, enable_logging=False):
    """
    Wraps a function and prints the shapes of all tensor inputs.
    Only prints when a new combination of shapes is seen.
    Thread-safe.

    Args:
        fn: Function to wrap
        enable_logging: Boolean flag to enable/disable logging
    """
    seen_shapes = set()

    def get_shape(obj):
        if isinstance(obj, torch.Tensor):
            return obj.shape
        elif isinstance(obj, (list, tuple)):
            if len(obj) > 1:
                return tuple(get_shape(x) for x in obj)
            return get_shape(obj[0])
        elif isinstance(obj, dict):
            return tuple(sorted((k, get_shape(v)) for k, v in obj.items()))
        else:
            return type(obj).__name__

    def wrapper(*args, **kwargs):
        shapes = tuple(get_shape(arg) for arg in args) + tuple(
            (k, get_shape(v))
            for k, v in kwargs.items()
            if isinstance(v, (torch.Tensor, list))
            and (len(keep_kwargs) > 0 and k in keep_kwargs)
        )
        if shapes not in seen_shapes:
            seen_shapes.add(shapes)
            if enable_logging:
                print(f"[ShapeLogger] New input shapes for {fn.__qualname__}: {shapes}")
        return fn(*args, **kwargs)

    # Allow toggling the flag at runtime
    wrapper.enable_logging = enable_logging

    def set_logging(enabled=False):
        nonlocal enable_logging
        enable_logging = enabled
        wrapper.enable_logging = enable_logging

    wrapper.set_logging = set_logging
    return wrapper
