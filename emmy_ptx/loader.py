"""Official deserialization and launch ABI checks. Only trusted IR is accepted."""
from dataclasses import dataclass, field
from math import prod
import json
from pathlib import Path
import re

from emmy.compiler.graph import Graph
from emmy.compiler.ir.base import ConstantOp, InputOp
from emmy.compiler.ir.kernel.ir import KernelOp

from .errors import CompileError


@dataclass(frozen=True)
class Kernel:
    name: str
    body: object
    arg_order: tuple[str, ...]
    shapes: dict[str, int]
    inputs: frozenset[str]
    outputs: frozenset[str]
    grid: tuple[int, ...]
    block: tuple[int, ...]
    scalar_outputs: frozenset[str] = frozenset()
    dimensions: dict[str, tuple[int, ...]] = field(default_factory=dict)
    smem_bytes: int = 0
    zero_outputs: frozenset[str] = frozenset()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CompileError(message)


def read_json(path: Path) -> dict:
    try:
        result = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise CompileError(f'{path.name}: cannot read JSON: {exc}') from exc
    require(isinstance(result, dict), f'{path.name}: expected a JSON object')
    return result


def load_kernel(path: Path) -> Kernel:
    raw = read_json(path)
    meta = read_json(path.with_name('meta.json'))
    # Emmy's body encoding is a repr string. Do not reimplement its deserializer.
    # Graph.from_dict uses eval internally: this is NOT an untrusted-input sandbox.
    try:
        graph = Graph.from_dict(raw)
    except Exception as exc:
        raise CompileError(f'invalid Emmy IR: {exc}') from exc
    kernels = [n for n in graph.nodes.values() if isinstance(n.op, KernelOp)]
    require(len(kernels) == 1, 'expected exactly one KernelOp')
    node = kernels[0]
    op = node.op
    require(isinstance(op.name, str) and re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', op.name), 'entry must be an ASCII identifier')
    require(not op.serial and not op.zero_delegated, 'serial axes and delegated zeroing are unsupported')
    graph_inputs, constants = set(), set()
    for n in graph.nodes.values():
        if n is node:
            continue
        require(isinstance(n.op, (InputOp, ConstantOp)), f'unsupported graph operation: {type(n.op).__name__}')
        require(not n.inputs, 'input and constant nodes must be leaves')
        buffers = {t.name for t in n.outputs}
        (constants if isinstance(n.op, ConstantOp) else graph_inputs).update(buffers)
    rows = meta.get('kernels')
    require(isinstance(rows, list) and all(isinstance(r, dict) for r in rows), 'meta.json: kernels must be a list of objects')
    rows = [r for r in rows if r.get('entry') == op.name and r.get('node') == node.id]
    require(len(rows) == 1, 'meta.json: expected one matching kernel entry and node')
    launch = rows[0]
    inputs = frozenset(node.inputs)
    outputs = frozenset(t.name for t in node.outputs)
    require(not inputs & outputs, 'in-place output buffers are unsupported')
    require(set(graph.inputs) == graph_inputs and inputs == graph_inputs | constants and set(graph.outputs) == outputs, 'graph input/output/constant contract does not match kernel')
    order = launch.get('arg_order')
    require(isinstance(order, list) and all(isinstance(x, str) for x in order), 'meta.json: invalid arg_order')
    require(len(set(order)) == len(order) and set(order) == inputs | outputs, 'meta.json: arg_order must contain each kernel buffer exactly once')
    require(all(re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', name) for name in order), 'buffer names must be ASCII identifiers')
    tensors = {}
    for n in graph.nodes.values():
        for tensor in n.outputs:
            require(tensor.name not in tensors, f'duplicate buffer: {tensor.name}')
            tensors[tensor.name] = tensor
    shapes, dimensions = {}, {}
    scalar_outputs = set()
    for name in order:
        require(name in tensors, f'unknown buffer: {name}')
        t = tensors[name]
        require(t.dtype.name == 'f32', f'unsupported dtype for {name}: {t.dtype.name}')
        require(all(d.is_static for d in t.shape), f'{name}: only static dimensions supported')
        dims = tuple(d.as_static() for d in t.shape)
        require(all(type(d) is int and d > 0 for d in dims), f'{name}: dimensions must be positive integers')
        dimensions[name] = dims
        if len(t.shape) == 0:
            require(name in outputs, f'{name}: rank-0 input buffers are unsupported')
            scalar_outputs.add(name)
            size = 1  # Official scalar outputs use a one-element pointer ABI.
        else:
            require(len(dims) <= 2, f'{name}: only rank-1 and rank-2 buffers supported')
            size = prod(dims)
        require(0 < size < 2**31, f'{name}: invalid or oversized buffer')
        shapes[name] = size
    args = launch.get('args')
    require(isinstance(args, list) and all(isinstance(a, dict) for a in args), 'meta.json: invalid args descriptors')
    require(all(isinstance(a.get('name'), str) for a in args), 'meta.json: argument names must be strings')
    require([a['name'] for a in args] == order, 'meta.json: args order must exactly match arg_order')
    for a in args:
        require(a.get('scalar', False) is False, 'meta.json: by-value scalar arguments are unsupported; buffers require pointers')
        require(type(a.get('output', False)) is bool and a.get('output', False) == (a['name'] in outputs), f'meta.json: output role mismatch for {a["name"]}')
        require(a.get('intermediate', False) is False, 'meta.json: intermediate arguments are unsupported')
        require(isinstance(a.get('shape'), list) and all(type(d) is int for d in a['shape']), 'meta.json: argument shape must contain integers')
        require(a.get('dtype') == 'f32' and a.get('shape') == list(dimensions[a['name']] or (1,)), f'meta.json: buffer descriptor mismatch for {a["name"]}')
        if a['name'] in constants:
            # Constants are runtime buffers. Values never enter generated PTX.
            data = a.get('data')
            require(isinstance(data, list) and len(data) == shapes[a['name']] and all(type(v) in (int, float) for v in data), f'meta.json: constant buffer {a["name"]} requires numeric data matching its shape')
            require('seed' not in a, 'meta.json: constant data and seed cannot both be supplied')
    geometry = []
    for key in ('grid', 'block'):
        dims = launch.get(key)
        require(isinstance(dims, list) and len(dims) == 3 and all(type(d) is int and d > 0 for d in dims), f'meta.json: invalid {key}')
        require(dims[1:] == [1, 1], f'meta.json: only 1-D {key} supported')
        geometry.append(tuple(dims))
    grid, block = geometry
    require(block[0] <= 1024 and grid[0] * block[0] < 2**31, 'unsupported launch size')
    smem_bytes = launch.get('smem_bytes', 0)
    require(type(smem_bytes) is int and 0 <= smem_bytes <= 99 * 1024, 'meta.json: invalid dynamic shared-memory byte count')
    zero_outputs = launch.get('zero_outputs', [])
    require(isinstance(zero_outputs, list) and all(isinstance(name, str) for name in zero_outputs)
            and len(zero_outputs) == len(set(zero_outputs)) and set(zero_outputs) <= outputs,
            'meta.json: zero_outputs must name distinct output buffers')
    return Kernel(op.name, op.body, tuple(order), shapes, inputs, outputs, grid, block, frozenset(scalar_outputs), dimensions, smem_bytes, frozenset(zero_outputs))
