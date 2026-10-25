"""Test-only cooperative model for the emitted sum/mean/row-reduction PTX subset.

Threads run until a shuffle/barrier rendezvous, then resume together. Shared
memory accesses are checked for initialization and cross-thread conflicts within
each block-barrier epoch. CTAs execute serially with fresh local state; global
loads/stores are counted across the launch and conflicting output owners fail.
This is deliberately not a PTX assembler, NVIDIA GPU emulator, proof of the
hardware memory model, or performance model.
"""
from dataclasses import dataclass
import re
import struct

import numpy as np

from ptx_model import input_buffers


@dataclass
class ReductionModelResult:
    outputs: dict[str, np.ndarray]
    store_counts: dict[str, np.ndarray]
    load_counts: dict[str, np.ndarray]
    barrier_count: int


def run_reduction_ptx(ptx, launch, overrides=None):
    """Interpret a one-dimensional grid of full-warp CTAs with private shared state."""
    grid, block = launch['grid'], launch['block']
    if (len(grid) != 3 or len(block) != 3 or grid[1:] != [1, 1]
            or block[1:] != [1, 1] or any(type(d) is not int or d <= 0 for d in grid + block)):
        raise AssertionError('reduction model requires a positive one-dimensional grid/block')
    threads = launch['block'][0]
    if not 32 <= threads <= 1024 or threads % 32:
        raise AssertionError('reduction model requires complete warps')
    required_threads = re.search(r'\.reqntid\s+(\d+)\s*,\s*(\d+)\s*,\s*(\d+)', ptx)
    if required_threads and list(map(int, required_threads.groups())) != launch['block']:
        raise AssertionError('required PTX block size differs from launch')
    buffers = input_buffers(launch, overrides)
    output_names = {arg['name'] for arg in launch['args'] if arg.get('output')}
    store_counts = {name: np.zeros(buffers[name].size, dtype=np.uint32) for name in output_names}
    load_counts = {name: np.zeros(buffer.size, dtype=np.uint32) for name, buffer in buffers.items()
                   if name not in output_names}
    bases = {arg['name']: (i + 1) << 40 for i, arg in enumerate(launch['args'])}
    parameters = re.findall(r'\.param\s+\.u64\s+(\w+)', ptx)
    if len(parameters) != len(launch['args']):
        raise AssertionError('parameter count differs from launch')
    params = {parameter: bases[arg['name']] for parameter, arg in zip(parameters, launch['args'])}

    register_types = {}
    for dtype, prefix, length in re.findall(r'\.reg\s+\.(\w+)\s+(%\w+)(?:<(\d+)>)?\s*;', ptx):
        if length:
            register_types.update((prefix + str(i), dtype) for i in range(int(length)))
        else:
            register_types[prefix] = dtype
    shared, shared_bases, initialized = {}, {}, {}
    for dtype, name, length in re.findall(r'\.shared\s+(?:\.align\s+\d+\s+)?\.(b8|f32)\s+(\w+)\[(\d+)\]\s*;', ptx):
        byte_count = int(length) * (4 if dtype == 'f32' else 1)
        if byte_count <= 0 or byte_count % 4 or name in shared:
            raise AssertionError('invalid modeled shared allocation')
        shared_bases[name] = 0x1000 + sum(a.size * 4 for a in shared.values())
        shared[name] = np.zeros(byte_count // 4, dtype=np.float32)
        initialized[name] = np.zeros(byte_count // 4, dtype=bool)

    body = ptx.split('{', 1)[1].rsplit('}', 1)[0]
    lines = [line.split('//', 1)[0].strip() for line in body.splitlines()]
    lines = [line for line in lines if line and not line.startswith('.')]
    labels = {line[:-1]: i for i, line in enumerate(lines) if line.endswith(':')}
    barriers = 0
    output_owners = {}
    step_limit = max(10000, sum(a.size for a in buffers.values()) * 64 // threads + 10000)

    def value(tid, operand):
        if operand.startswith('%'):
            if operand not in registers[tid]:
                raise AssertionError(f'uninitialized modeled register {operand}, thread {tid}')
            return registers[tid][operand]
        if operand in shared_bases:
            return shared_bases[operand]
        if operand.startswith('0f'):
            return np.float32(struct.unpack('!f', bytes.fromhex(operand[2:]))[0])
        return int(operand, 0)

    def address(tid, operand):
        if not operand.startswith('[') or not operand.endswith(']'):
            raise AssertionError('malformed modeled memory operand')
        return sum(int(value(tid, part.strip())) for part in operand[1:-1].split('+'))

    def locate(addr, spaces, arrays):
        for name, base in spaces.items():
            offset = addr - base
            if 0 <= offset < arrays[name].size * 4:
                if offset % 4:
                    raise AssertionError('unaligned modeled f32 access')
                return name, offset // 4
        raise AssertionError(f'out-of-bounds modeled memory access: {addr}')

    def shared_access(tid, location, writing):
        name, offset = location
        prior_writer = writers.get(location)
        prior_readers = readers.get(location, set())
        if prior_writer is not None and prior_writer != tid:
            raise AssertionError('unsynchronized cross-thread shared memory conflict')
        if writing:
            if prior_readers - {tid}:
                raise AssertionError('unsynchronized cross-thread shared read/write conflict')
            writers[location] = tid
            initialized[name][offset] = True
        else:
            if not initialized[name][offset]:
                raise AssertionError('read of uninitialized shared memory')
            readers.setdefault(location, set()).add(tid)

    def run_thread(tid):
        reg = registers[tid]
        while not done[tid] and waiting[tid] is None:
            if pcs[tid] >= len(lines):
                raise AssertionError('modeled thread fell off the kernel')
            steps[tid] += 1
            if steps[tid] > step_limit:
                raise AssertionError('modeled thread exceeded bounded instruction count')
            pc = pcs[tid]
            line = lines[pc]
            pcs[tid] += 1
            if line.endswith(':'):
                continue
            if line.startswith('@'):
                predicate, line = line.split(None, 1)
                negate = predicate.startswith('@!')
                enabled = bool(value(tid, predicate[2:] if negate else predicate[1:]))
                if enabled == negate:
                    continue
            opcode, _, operands = line.removesuffix(';').partition(' ')
            args = [part.strip() for part in operands.split(',')]
            if opcode == 'ret':
                done[tid] = True
            elif opcode == 'bra':
                if args[0] not in labels:
                    raise AssertionError('unknown modeled branch target')
                pcs[tid] = labels[args[0]]
            elif opcode == 'ld.param.u64':
                reg[args[0]] = params[args[1][1:-1]]
            elif opcode in ('mov.u32', 'mov.u64', 'cvta.to.global.u64'):
                reg[args[0]] = int(value(tid, args[1]))
            elif opcode == 'mov.f32':
                reg[args[0]] = np.float32(value(tid, args[1]))
            elif opcode == 'mov.b32':
                source = value(tid, args[1])
                if register_types.get(args[0]) == 'f32':
                    reg[args[0]] = np.uint32(int(source) & 0xffffffff).view(np.float32)
                elif isinstance(source, np.floating):
                    reg[args[0]] = int(np.float32(source).view(np.uint32))
                else:
                    reg[args[0]] = int(source) & 0xffffffff
            elif opcode in ('add.u32', 'add.u64', 'mul.lo.u32', 'mul.wide.u32'):
                a, b = int(value(tid, args[1])), int(value(tid, args[2]))
                result = a + b if opcode.startswith('add') else a * b
                reg[args[0]] = result & (0xffffffff if opcode in ('add.u32', 'mul.lo.u32') else 0xffffffffffffffff)
            elif opcode in ('and.b32', 'shr.u32'):
                a, b = int(value(tid, args[1])), int(value(tid, args[2]))
                reg[args[0]] = (a & b) if opcode == 'and.b32' else (a & 0xffffffff) >> b
            elif opcode in ('setp.ge.u32', 'setp.ne.u32', 'setp.eq.u32', 'setp.lt.u32'):
                a, b = int(value(tid, args[1])), int(value(tid, args[2]))
                reg[args[0]] = {'ge': a >= b, 'ne': a != b, 'eq': a == b, 'lt': a < b}[opcode.split('.')[1]]
            elif opcode == 'ld.global.f32':
                name, offset = locate(address(tid, args[1]), bases, buffers)
                if name not in load_counts:
                    raise AssertionError('modeled reduction may only load input buffers')
                load_counts[name][offset] += 1
                reg[args[0]] = buffers[name].flat[offset]
            elif opcode == 'st.global.f32':
                name, offset = locate(address(tid, args[0]), bases, buffers)
                if name not in output_names:
                    raise AssertionError('modeled write to an input buffer')
                owner = (cta, tid)
                if (name, offset) in output_owners and output_owners[(name, offset)] != owner:
                    raise AssertionError('multiple threads or CTAs write one output element')
                output_owners[(name, offset)] = owner
                buffers[name].flat[offset] = np.float32(value(tid, args[1]))
                store_counts[name][offset] += 1
            elif opcode in ('ld.shared.f32', 'st.shared.f32'):
                writing = opcode.startswith('st.')
                location = locate(address(tid, args[0] if writing else args[1]), shared_bases, shared)
                shared_access(tid, location, writing)
                name, offset = location
                if writing:
                    shared[name][offset] = np.float32(value(tid, args[1]))
                else:
                    reg[args[0]] = shared[name][offset]
            elif opcode in ('add.rn.f32', 'div.rn.f32'):
                operation = np.add if opcode == 'add.rn.f32' else np.divide
                reg[args[0]] = operation(np.float32(value(tid, args[1])), np.float32(value(tid, args[2])), dtype=np.float32)
            elif opcode == 'bar.sync':
                if len(args) != 1:
                    raise AssertionError('only whole-CTA barriers modeled')
                waiting[tid] = ('barrier', pc, int(value(tid, args[0])))
            elif opcode == 'shfl.sync.bfly.b32':
                if len(args) != 5:
                    raise AssertionError('unsupported shuffle operands')
                delta, control, mask = (int(value(tid, arg)) for arg in args[2:])
                if not 0 <= delta < 32 or control != 31 or mask != 0xffffffff:
                    raise AssertionError('only full-warp butterfly shuffle modeled')
                waiting[tid] = ('shuffle', pc, args[0], int(value(tid, args[1])), delta, control, mask)
            else:
                raise AssertionError(f'unsupported modeled instruction: {line}')

    with np.errstate(over='ignore', invalid='ignore', under='ignore', divide='ignore'):
        for cta in range(grid[0]):
            # Shared storage, its initialization proof and rendezvous state belong
            # to one CTA. A preceding row must never initialize a following row.
            for name, data in shared.items():
                data.fill(0)
                initialized[name].fill(False)
            readers, writers = {}, {}
            registers = [{'%tid.x': tid, '%ctaid.x': cta, '%ntid.x': threads} for tid in range(threads)]
            pcs, steps = [0] * threads, [0] * threads
            done, waiting = [False] * threads, [None] * threads
            while not all(done):
                progressed = False
                for tid in range(threads):
                    if not done[tid] and waiting[tid] is None:
                        run_thread(tid)
                        progressed = True
                for start in range(0, threads, 32):
                    warp = waiting[start:start + 32]
                    if all(event is not None and event[0] == 'shuffle' for event in warp):
                        contracts = {(event[1], *event[4:]) for event in warp}
                        if len(contracts) != 1:
                            raise AssertionError('divergent modeled warp shuffle rendezvous')
                        for lane, event in enumerate(warp):
                            registers[start + lane][event[2]] = warp[lane ^ event[4]][3]
                            waiting[start + lane] = None
                        progressed = True
                if all(event is not None and event[0] == 'barrier' for event in waiting):
                    if len(set(waiting)) != 1:
                        raise AssertionError('divergent modeled block barrier rendezvous')
                    waiting[:] = [None] * threads
                    readers.clear()
                    writers.clear()
                    barriers += 1
                    progressed = True
                if not progressed:
                    raise AssertionError('deadlock: incomplete shuffle or block barrier participation')
    for name, counts in store_counts.items():
        if not np.all(counts == 1):
            raise AssertionError(f'output {name} must be written exactly once')
    return ReductionModelResult({name: buffers[name] for name in output_names}, store_counts, load_counts, barriers)
