"""Official deserialization and launch ABI checks. Only trusted IR is accepted."""
from dataclasses import dataclass
import json
from pathlib import Path
import re

from emmy.compiler.graph import Graph
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
    for n in graph.nodes.values():
        require(n is node or type(n.op).__name__ == 'InputOp', f'unsupported graph operation: {type(n.op).__name__}')
    rows = meta.get('kernels')
    require(isinstance(rows, list) and all(isinstance(r, dict) for r in rows), 'meta.json: kernels must be a list of objects')
    rows = [r for r in rows if r.get('entry') == op.name and r.get('node') == node.id]
    require(len(rows) == 1, 'meta.json: expected one matching kernel entry and node')
    launch = rows[0]
    inputs = frozenset(node.inputs)
    outputs = frozenset(t.name for t in node.outputs)
    require(not inputs & outputs, 'in-place output buffers are unsupported')
    require(set(graph.inputs) == inputs and set(graph.outputs) == outputs, 'graph input/output contract does not match kernel')
    order = launch.get('arg_order')
    require(isinstance(order, list) and all(isinstance(x, str) for x in order), 'meta.json: invalid arg_order')
    require(len(set(order)) == len(order) and set(order) == inputs | outputs, 'meta.json: arg_order must contain each kernel buffer exactly once')
    require(all(re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', name) for name in order), 'buffer names must be ASCII identifiers')
    tensors = {}
    for n in graph.nodes.values():
        for tensor in n.outputs:
            require(tensor.name not in tensors, f'duplicate buffer: {tensor.name}')
            tensors[tensor.name] = tensor
    shapes = {}
    for name in order:
        require(name in tensors, f'unknown buffer: {name}')
        t = tensors[name]
        require(t.dtype.name == 'f32', f'unsupported dtype for {name}: {t.dtype.name}')
        require(len(t.shape) == 1 and t.shape[0].is_static, f'{name}: only static 1-D buffers supported')
        size = t.shape[0].as_static()
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
        require(a.get('dtype') == 'f32' and a.get('shape') == [shapes[a['name']]], f'meta.json: buffer descriptor mismatch for {a["name"]}')
    geometry = []
    for key in ('grid', 'block'):
        dims = launch.get(key)
        require(isinstance(dims, list) and len(dims) == 3 and all(type(d) is int and d > 0 for d in dims), f'meta.json: invalid {key}')
        require(dims[1:] == [1, 1], f'meta.json: only 1-D {key} supported')
        geometry.append(tuple(dims))
    grid, block = geometry
    require(block[0] <= 1024 and grid[0] * block[0] < 2**31, 'unsupported launch size')
    require(launch.get('smem_bytes', 0) == 0 and not launch.get('zero_outputs', []), 'shared memory and zero_outputs are unsupported')
    return Kernel(op.name, op.body, tuple(order), shapes, inputs, outputs, grid, block)
