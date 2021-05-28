import numpy as np
import os
import subprocess
import heapq as hq
from .core import Function, Variable


def _dot_var(v: Variable, verbose=False):
    name = v.name
    if verbose and v.data:
        if name:
            name += ": "
        name += str(v.shape) + " " + str(v.dtype)
    return f'{id(v)} [label="{name}", color=orange, style=filled]\n'


def _dot_func(f: Function):
    txt = f'{id(f)} [label="{f.__class__.__name__}", ' \
        'color=lightblue, style=filled, shape=box]\n'
    for x in f.inputs:
        txt += f'{id(x)} -> {id(f)}\n'
    for y in f.outputs:
        txt += f'{id(f)} -> {id(y())}\n'
    return txt


def get_dot_graph(output: Variable, verbose=False):
    txt = ''
    funcs: list[tuple[int, Function]] = []
    seen_set = set()

    def add_func(f: Function):
        if f not in seen_set:
            hq.heappush(funcs, (-f.generation, f))
            seen_set.add(f)
    add_func(output.creator)
    txt += _dot_var(output, verbose)

    while funcs:
        _, func = hq.heappop(funcs)
        txt += _dot_func(func)
        for x in func.inputs:
            txt += _dot_var(x, verbose)

            if x.creator:
                add_func(x.creator)
    return f'digraph g {{\n{txt}}} '


def plot_dot_graph(output: Variable, verbose=False, to_file="graph.png"):
    dot_graph = get_dot_graph(output, verbose)
    ext = os.path.splitext(to_file)[1][1:]

    # return subprocess.run(["cat"], input=dot_graph, text=True)
    return subprocess.run(["dot", "-T", ext, "-o", to_file],
                          input=dot_graph, text=True)


def sum_to(x: np.ndarray, shape: tuple[int, ...]):
    ndim = len(shape)
    lead = x.ndim - ndim
    lead_axis = tuple(range(lead))

    axis = tuple([i + lead for i, sx in enumerate(shape) if sx == 1])
    y = x.sum(lead_axis + axis, keepdims=True)
    if lead > 0:
        y = y.squeeze(lead_axis)
    return y
