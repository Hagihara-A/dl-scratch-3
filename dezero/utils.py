from __future__ import annotations
from typing import Optional
import numpy as np
import os
import subprocess
import heapq as hq
from .core import Function, Variable
import urllib


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


def reshape_sum_backward(gy: Variable, x_shape: tuple[int, ...],
                         axis: Optional[int | tuple[int]], keepdims: bool):
    """Reshape gradient appropriately for dezero.functions.sum's backward.
    Args:
        gy (dezero.Variable): Gradient variable from the output by backprop.
        x_shape (tuple): Shape used at sum function's forward.
        axis (None or int or tuple of ints): Axis used at sum function's
            forward.
        keepdims (bool): Keepdims used at sum function's forward.
    Returns:
        dezero.Variable: Gradient variable which is reshaped appropriately
    """
    ndim = len(x_shape)
    tupled_axis = axis
    if axis is None:
        tupled_axis = None
    elif not isinstance(axis, tuple):
        tupled_axis = (axis,)

    if not (ndim == 0 or tupled_axis is None or keepdims):
        actual_axis = [a if a >= 0 else a + ndim for a in tupled_axis]
        shape = list(gy.shape)
        for a in sorted(actual_axis):
            shape.insert(a, 1)
    else:
        shape = gy.shape

    gy = gy.reshape(shape)  # reshape
    return gy


def to_onehot(t: np.ndarray, labels: int):
    if (t.ndim == 1):
        onehot = np.zeros((len(t), labels))
        onehot[np.arange(len(t)), t] = 1
        return onehot
    else:
        raise ValueError("ndim must be 1")


def pair(x):
    if isinstance(x, int):
        return (x, x)
    elif isinstance(x, tuple):
        assert len(x) == 2
        return x
    else:
        raise ValueError


def show_progress(block_num, block_size, total_size):
    bar_template = "\r[{}] {:.2f}%"

    downloaded = block_num * block_size
    p = downloaded / total_size * 100
    i = int(downloaded / total_size * 30)
    if p >= 100.0:
        p = 100.0
    if i >= 30:
        i = 30
    bar = "#" * i + "." * (30 - i)
    print(bar_template.format(bar, p), end='')


cache_dir = os.path.join(os.path.expanduser('~'), '.dezero')


def get_file(url, file_name=None):
    """Download a file from the `url` if it is not in the cache.
    The file at the `url` is downloaded to the `~/.dezero`.
    Args:
        url (str): URL of the file.
        file_name (str): Name of the file. It `None` is specified the original
            file name is used.
    Returns:
        str: Absolute path to the saved file.
    """
    if file_name is None:
        file_name = url[url.rfind('/') + 1:]
    file_path = os.path.join(cache_dir, file_name)

    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)

    if os.path.exists(file_path):
        return file_path

    print("Downloading: " + file_name)
    try:
        urllib.request.urlretrieve(url, file_path, show_progress)
    except (Exception, KeyboardInterrupt):
        if os.path.exists(file_path):
            os.remove(file_path)
        raise
    print(" Done")

    return file_path
