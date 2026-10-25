"""A deliberately small CPU model for our emitted pointwise PTX subset.

This is a test oracle for address/operand flow, not NVIDIA PTX assembly or GPU
execution. It rejects instructions outside the model and checks every memory
access plus full output coverage. Float arithmetic uses NumPy float32 rounding.
"""
from dataclasses import dataclass
import re
import struct

import numpy as np


@dataclass
class ModelResult:
    outputs: dict[str, np.ndarray]
    store_counts: dict[str, np.ndarray]


def input_buffers(launch, overrides=None):
    """Build f32 buffers using the official metadata's seed/data descriptions."""
    overrides = overrides or {}
    buffers = {}
    for arg in launch['args']:
        name, shape = arg['name'], tuple(arg['shape'])
        if name in overrides:
            value = np.asarray(overrides[name], dtype=np.float32)
            if value.shape != shape:
                raise AssertionError(f'override shape mismatch for {name}')
            buffers[name] = value.copy()
        elif arg.get('output'):
            buffers[name] = np.zeros(shape, dtype=np.float32)
        elif 'data' in arg:
            buffers[name] = np.asarray(arg['data'], dtype=np.float32).reshape(shape)
        else:
            buffers[name] = np.random.default_rng(arg['seed']).standard_normal(shape).astype(np.float32)
    return buffers


def run_ptx(ptx, launch, overrides=None):
    """Interpret all launched threads and return outputs and per-element writes."""
    buffers = input_buffers(launch, overrides)
    parameters = re.findall(r'\.param\s+\.u64\s+(\w+)', ptx)
    if len(parameters) != len(launch['args']):
        raise AssertionError('parameter count differs from launch')
    bases = {arg['name']: (i + 1) << 40 for i, arg in enumerate(launch['args'])}
    params = {parameter: bases[arg['name']] for parameter, arg in zip(parameters, launch['args'])}
    outputs = {a['name'] for a in launch['args'] if a.get('output')}
    counts = {name: np.zeros(buffers[name].size, dtype=np.uint16) for name in outputs}
    body = ptx.split('{', 1)[1].rsplit('}', 1)[0]
    lines = [line.strip() for line in body.splitlines() if line.strip() and not line.strip().startswith(('.', '//'))]
    labels = {line[:-1]: i for i, line in enumerate(lines) if line.endswith(':')}

    def location(address, writing=False):
        for name, base in bases.items():
            delta = address - base
            if 0 <= delta < buffers[name].size * 4:
                if delta % 4:
                    raise AssertionError('unaligned modeled f32 access')
                if writing and name not in outputs:
                    raise AssertionError(f'write to input {name}')
                return name, delta // 4
        raise AssertionError(f'out-of-bounds modeled address: {address}')

    for cta in range(launch['grid'][0]):
        for tid in range(launch['block'][0]):
            registers = {'%tid.x': tid, '%ctaid.x': cta, '%ntid.x': launch['block'][0]}

            def value(operand):
                if operand.startswith('%'):
                    return registers[operand]
                if operand.startswith('0f'):
                    return np.float32(struct.unpack('!f', bytes.fromhex(operand[2:]))[0])
                return int(operand)

            def address(operand):
                inside = operand.removeprefix('[').removesuffix(']')
                parts = inside.split('+')
                return sum(value(part.strip()) for part in parts)

            pc = 0
            steps = 0
            with np.errstate(over='ignore', invalid='ignore', under='ignore'):
                while pc < len(lines):
                    steps += 1
                    if steps > len(lines) * 2:
                        raise AssertionError('unexpected loop in pointwise PTX model')
                    line = lines[pc]
                    pc += 1
                    if line.endswith(':'):
                        continue
                    if line.startswith('@'):
                        predicate, line = line.split(' ', 1)
                        negate = predicate.startswith('@!')
                        enabled = bool(value(predicate[2:] if negate else predicate[1:]))
                        if enabled == negate:
                            continue
                    opcode, _, operands = line.removesuffix(';').partition(' ')
                    args = [part.strip() for part in operands.split(',')]
                    if opcode == 'ret':
                        break
                    if opcode == 'bra':
                        pc = labels[args[0]]
                        continue
                    if opcode == 'ld.param.u64':
                        registers[args[0]] = params[args[1][1:-1]]
                    elif opcode in ('mov.u32', 'cvta.to.global.u64'):
                        registers[args[0]] = value(args[1])
                    elif opcode in ('add.u32', 'mul.lo.u32', 'mad.lo.u32', 'mul.wide.u32', 'add.u64'):
                        a, b = value(args[1]), value(args[2])
                        result = a + b if opcode.startswith('add') else a * b
                        if opcode.startswith('mad'):
                            result += value(args[3])
                        if opcode in ('add.u32', 'mul.lo.u32', 'mad.lo.u32'):
                            result &= 0xffffffff
                        registers[args[0]] = result
                    elif opcode in ('setp.ge.u32', 'setp.lt.f32'):
                        a, b = value(args[1]), value(args[2])
                        registers[args[0]] = a >= b if opcode == 'setp.ge.u32' else a < b
                    elif opcode == 'selp.f32':
                        registers[args[0]] = value(args[1] if value(args[3]) else args[2])
                    elif opcode == 'ld.global.f32':
                        name, offset = location(address(args[1]))
                        registers[args[0]] = buffers[name].flat[offset]
                    elif opcode in ('add.rn.f32', 'mul.rn.f32'):
                        a, b = np.float32(value(args[1])), np.float32(value(args[2]))
                        registers[args[0]] = np.add(a, b, dtype=np.float32) if opcode.startswith('add') else np.multiply(a, b, dtype=np.float32)
                    elif opcode == 'max.NaN.f32':
                        a, b = np.float32(value(args[1])), np.float32(value(args[2]))
                        if np.isnan(a) or np.isnan(b):
                            result = np.float32(np.nan)
                        elif a == 0 and b == 0:
                            result = np.float32(0)
                        else:
                            result = np.maximum(a, b)
                        registers[args[0]] = result
                    elif opcode == 'st.global.f32':
                        name, offset = location(address(args[0]), writing=True)
                        buffers[name].flat[offset] = value(args[1])
                        counts[name][offset] += 1
                    else:
                        raise AssertionError(f'unsupported modeled instruction: {line}')
    for name, writes in counts.items():
        if not np.all(writes == 1):
            raise AssertionError(f'output {name} must be written exactly once per element')
    return ModelResult({name: buffers[name] for name in outputs}, counts)
