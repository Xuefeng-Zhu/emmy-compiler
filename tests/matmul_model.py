"""Independent CPU operand/control-flow model of emitted scalar matmul PTX.

This is not an assembler or a GPU emulator. It runs a deliberately small PTX
subset with NumPy arrays representing independent threads. No IR is consulted.
Shared memory is private to each CTA. Async copies are queued until commit/wait;
barriers and shared access epochs check initialization and cross-thread ordering.
Atomics execute one deterministic legal CTA order with f32 rounding/FTZ. This
model cannot establish PTX assembler legality, hardware races, or performance.
"""
from dataclasses import dataclass
import re
import struct

import numpy as np

from ptx_model import input_buffers


@dataclass
class MatmulResult:
    outputs: dict[str, np.ndarray]
    store_counts: dict[str, np.ndarray]
    writer_threads: dict[str, np.ndarray]
    barrier_count: int
    async_copy_bytes: int
    committed_groups: int
    completed_groups: int


def run_matmul_ptx(ptx, launch, overrides=None, *, max_steps=2_000_000):
    """Execute emitted instructions, checking ABI, memory and output ownership."""
    required = re.search(r'\.reqntid\s+(\d+)\s*,\s*(\d+)\s*,\s*(\d+)', ptx)
    if required and list(map(int, required.groups())) != launch['block']:
        raise AssertionError('required block size differs from launch')
    buffers = input_buffers(launch, overrides)
    outputs = {arg['name'] for arg in launch['args'] if arg.get('output')}
    counts = {name: np.zeros(buffers[name].size, np.uint32) for name in outputs}
    writers = {name: np.full(buffers[name].size, -1, np.int64) for name in outputs}
    parameters = re.findall(r'\.param\s+\.u64\s+(\w+)', ptx)
    if len(parameters) != len(launch['args']) or len(set(parameters)) != len(parameters):
        raise AssertionError('parameter count or names do not match launch')
    # Bind by argument position, independently of parameter spelling.
    bases = {arg['name']: (i + 1) << 40 for i, arg in enumerate(launch['args'])}
    params = dict(zip(parameters, (bases[arg['name']] for arg in launch['args'])))
    text = ptx.split('{', 1)[1].rsplit('}', 1)[0]
    lines = [line.split('//', 1)[0].strip() for line in text.splitlines()]
    lines = [line for line in lines if line and not line.startswith('.')]
    labels = {}
    instructions = []
    for line in lines:
        if line.endswith(':'):
            if line[:-1] in labels:
                raise AssertionError('duplicate PTX label')
            labels[line[:-1]] = len(instructions)
            continue
        predicate = None
        if line.startswith('@'):
            predicate, line = line.split(None, 1)
        if not line.endswith(';'):
            raise AssertionError('instruction must end in semicolon')
        opcode, _, operands = line[:-1].partition(' ')
        instructions.append((predicate, opcode, tuple(x.strip() for x in operands.split(','))))

    block_size = launch['block'][0]
    nctas = launch['grid'][0]
    if launch['grid'][1:] != [1, 1] or launch['block'][1:] != [1, 1]:
        raise AssertionError('model requires a one-dimensional launch')
    shared_bases = {}
    shared_bytes = 0
    for align, name, size in re.findall(r'(?<!extern )\.shared\s+\.align\s+(\d+)\s+\.b8\s+(\w+)\[(\d+)\]', ptx):
        alignment = int(align)
        shared_bytes = (shared_bytes + alignment - 1) // alignment * alignment
        shared_bases[name] = 0x1000 + shared_bytes
        shared_bytes += int(size)
    dynamic = re.findall(r'\.extern\s+\.shared\s+\.align\s+(\d+)\s+\.b8\s+(\w+)\[\]', ptx)
    if len(dynamic) > 1:
        raise AssertionError('one dynamic shared pool supported')
    if dynamic:
        alignment, name = dynamic[0]
        alignment = int(alignment)
        shared_bytes = (shared_bytes + alignment - 1) // alignment * alignment
        shared_bases[name] = 0x1000 + shared_bytes
        shared_bytes += launch.get('smem_bytes', 0)
    if shared_bytes <= 0 or shared_bytes % 4:
        raise AssertionError('missing or unaligned shared allocation')
    shared_words = shared_bytes // 4
    shared_memory = np.zeros(nctas * shared_words, np.float32)
    shared_initialized = np.zeros(shared_memory.size, bool)
    write_epoch = np.full(shared_memory.size, -1, np.int64)
    write_thread = np.full(shared_memory.size, -1, np.int64)
    read_epoch = np.full(shared_memory.size, -1, np.int64)
    pending = np.zeros(shared_memory.size, np.int32)
    epoch = 0
    barrier_count = async_copy_bytes = committed_groups = completed_groups = 0
    current_copies, groups = [], []
    nthreads = launch['grid'][0] * launch['block'][0]
    gids = np.arange(nthreads, dtype=np.int64)
    registers = {'%tid.x': gids % launch['block'][0],
                 '%ctaid.x': gids // launch['block'][0],
                 '%ntid.x': np.full(nthreads, launch['block'][0], np.int64)}
    initialized = {name: np.ones(nthreads, bool) for name in registers}
    pc = np.zeros(nthreads, np.int64)
    alive = np.ones(nthreads, bool)
    immediates = {}

    def value(operand, mask):
        if operand.startswith('%'):
            if operand not in registers or not np.all(initialized[operand][mask]):
                raise AssertionError('read of uninitialized register: ' + operand)
            return registers[operand][mask]
        if operand in shared_bases:
            return shared_bases[operand]
        if operand not in immediates:
            if re.fullmatch(r'0[fF][0-9a-fA-F]{8}', operand):
                immediates[operand] = np.float32(struct.unpack('!f', bytes.fromhex(operand[2:]))[0])
            else:
                immediates[operand] = int(operand, 0)
        return immediates[operand]

    def assign(name, result, mask):
        result = np.asarray(result)
        dtype = np.float32 if result.dtype.kind == 'f' else (bool if result.dtype.kind == 'b' else np.int64)
        if name not in registers:
            registers[name] = np.zeros(nthreads, dtype=dtype)
            initialized[name] = np.zeros(nthreads, bool)
        if registers[name].dtype.kind != np.dtype(dtype).kind:
            raise AssertionError('modeled register type changed: ' + name)
        registers[name][mask] = result
        initialized[name][mask] = True

    def signed(values):
        values = np.asarray(values, np.int64) & 0xffffffff
        return np.where(values >= 2**31, values - 2**32, values)

    def address(operand, mask):
        if not (operand.startswith('[') and operand.endswith(']')):
            raise AssertionError('invalid memory operand')
        return sum(value(part.strip(), mask) for part in operand[1:-1].split('+'))

    def memory(addresses, mask, writing=None, atomic=False):
        addresses = np.broadcast_to(np.asarray(addresses, np.int64), (np.count_nonzero(mask),))
        matched = np.zeros(addresses.size, bool)
        loaded = np.empty(addresses.size, np.float32)
        for name, base in bases.items():
            offsets = addresses - base
            belongs = (offsets >= 0) & (offsets < buffers[name].size * 4)
            if not np.any(belongs):
                continue
            if np.any(offsets[belongs] % 4):
                raise AssertionError('unaligned modeled memory access')
            indices = offsets[belongs] // 4
            matched |= belongs
            if writing is None:
                if name in outputs:
                    raise AssertionError('read of output-only buffer')
                loaded[belongs] = buffers[name].reshape(-1)[indices]
            else:
                if name not in outputs:
                    raise AssertionError('write to input-only buffer')
                data = np.broadcast_to(np.asarray(writing, np.float32), addresses.shape)
                np.add.at(counts[name], indices, 1)
                owner_ids = np.flatnonzero(mask)[belongs]
                writers[name][indices] = owner_ids
                flat = buffers[name].reshape(-1)
                if atomic:
                    if name not in launch.get('zero_outputs', []):
                        raise AssertionError('atomic output lacks zero_outputs contract')
                    ctas = owner_ids // block_size
                    for cta in np.unique(ctas):
                        selected = ctas == cta
                        target, addend = indices[selected], data[belongs][selected]
                        if np.unique(target).size != target.size:
                            raise AssertionError('duplicate atomic destinations within one CTA unsupported by model')
                        tiny = np.finfo(np.float32).tiny
                        left = flat[target]
                        left = np.where(np.abs(left) < tiny, np.copysign(np.float32(0), left), left)
                        addend = np.where(np.abs(addend) < tiny, np.copysign(np.float32(0), addend), addend)
                        result = np.add(left, addend, dtype=np.float32)
                        flat[target] = np.where(np.abs(result) < tiny, np.copysign(np.float32(0), result), result)
                else:
                    flat[indices] = data[belongs]
        if not np.all(matched):
            raise AssertionError('out-of-bounds modeled memory access')
        return loaded

    def shared_locations(addresses, mask):
        ids = np.flatnonzero(mask)
        addresses = np.broadcast_to(np.asarray(addresses, np.int64), ids.shape)
        offset = addresses - 0x1000
        if np.any(offset < 0) or np.any(offset >= shared_bytes) or np.any(offset % 4):
            raise AssertionError('out-of-bounds or unaligned shared access')
        return (ids // block_size) * shared_words + offset // 4, ids

    def reserve_write(indices, ids):
        if np.any((write_epoch[indices] == epoch) & (write_thread[indices] != ids)):
            raise AssertionError('unsynchronized cross-thread shared write conflict')
        if np.any(read_epoch[indices] == epoch):
            raise AssertionError('shared write before readers synchronize')
        if np.unique(indices).size != indices.size:
            raise AssertionError('multiple modeled threads write one shared slot')
        write_epoch[indices] = epoch
        write_thread[indices] = ids

    def finish_group(group):
        for indices, ids, data in group:
            pending[indices] -= 1
            shared_memory[indices] = data
            shared_initialized[indices] = True
            # A wait establishes completion for the issuing thread; peers still
            # require a following CTA barrier before reading that data.
            write_epoch[indices] = epoch
            write_thread[indices] = ids

    def full_ctas(mask):
        participants = mask.reshape(nctas, block_size)
        if not np.all(np.all(participants, axis=1)):
            raise AssertionError('divergent CTA rendezvous or prematurely returned thread')

    steps = 0
    with np.errstate(all='ignore'):
        while np.any(alive):
            steps += 1
            if steps > max_steps:
                raise AssertionError('bounded CPU model step limit exceeded')
            current = np.min(pc[alive])
            if not 0 <= current < len(instructions):
                raise AssertionError('thread fell off kernel without ret')
            mask = alive & (pc == current)
            pc[mask] += 1
            predicate, opcode, args = instructions[current]
            if predicate:
                enabled = np.asarray(value(predicate[2:] if predicate.startswith('@!') else predicate[1:], mask), bool)
                if predicate.startswith('@!'):
                    enabled = ~enabled
                enabled_mask = np.zeros(nthreads, bool)
                enabled_mask[mask] = enabled
                mask = enabled_mask
                if not np.any(mask):
                    continue
            if opcode == 'ret':
                alive[mask] = False
                continue
            if opcode == 'bra':
                if args[0] not in labels:
                    raise AssertionError('branch to unknown label')
                pc[mask] = labels[args[0]]
                continue
            if opcode == 'ld.param.u64':
                assign(args[0], params[args[1][1:-1]], mask)
            elif opcode in ('mov.u32', 'mov.s32', 'mov.u64', 'mov.b32', 'cvta.to.global.u64'):
                assign(args[0], value(args[1], mask), mask)
            elif opcode == 'mov.f32':
                assign(args[0], np.asarray(value(args[1], mask), np.float32), mask)
            elif opcode == 'ld.global.f32':
                assign(args[0], memory(address(args[1], mask), mask), mask)
            elif opcode == 'st.global.f32':
                memory(address(args[0], mask), mask, value(args[1], mask))
            elif opcode == 'red.global.add.f32':
                memory(address(args[0], mask), mask, value(args[1], mask), atomic=True)
            elif opcode == 'ld.shared.f32':
                indices, ids = shared_locations(address(args[1], mask), mask)
                if np.any(~shared_initialized[indices]) or np.any(pending[indices]):
                    raise AssertionError('shared read before async completion or initialization')
                if np.any((write_epoch[indices] == epoch) & (write_thread[indices] != ids)):
                    raise AssertionError('shared read before CTA synchronization')
                read_epoch[indices] = epoch
                assign(args[0], shared_memory[indices], mask)
            elif opcode == 'st.shared.f32':
                indices, ids = shared_locations(address(args[0], mask), mask)
                reserve_write(indices, ids)
                shared_memory[indices] = value(args[1], mask)
                shared_initialized[indices] = True
            elif opcode in ('cp.async.cg.shared.global', 'cp.async.ca.shared.global'):
                width = int(args[2])
                if width not in (4, 8, 16):
                    raise AssertionError('invalid modeled cp.async width')
                shared_address = address(args[0], mask)
                global_address = address(args[1], mask)
                if np.any(np.asarray(shared_address) % width) or np.any(np.asarray(global_address) % width):
                    raise AssertionError('unaligned modeled cp.async')
                for offset in range(0, width, 4):
                    indices, ids = shared_locations(shared_address + offset, mask)
                    if np.any(pending[indices]):
                        raise AssertionError('overlapping pending async copies')
                    reserve_write(indices, ids)
                    pending[indices] += 1
                    data = memory(global_address + offset, mask).copy()
                    current_copies.append((indices, ids, data))
                async_copy_bytes += width * np.count_nonzero(mask)
            elif opcode == 'cp.async.commit_group':
                full_ctas(mask)
                groups.append(current_copies)
                current_copies = []
                committed_groups += 1
            elif opcode == 'cp.async.wait_group':
                full_ctas(mask)
                remain = int(args[0])
                if not 0 <= remain <= 7:
                    raise AssertionError('invalid cp.async wait group')
                while len(groups) > remain:
                    finish_group(groups.pop(0))
                    completed_groups += 1
            elif opcode == 'bar.sync':
                if args != ('0',):
                    raise AssertionError('only whole-CTA barrier zero modeled')
                full_ctas(mask)
                epoch += 1
                barrier_count += 1
            elif re.fullmatch(r'setp\.(eq|ne|lt|le|gt|ge)\.(s32|u32|f32)', opcode):
                _, relation, dtype = opcode.split('.')
                a, b = value(args[1], mask), value(args[2], mask)
                if dtype == 's32':
                    a, b = signed(a), signed(b)
                elif dtype == 'u32':
                    a, b = np.asarray(a, np.int64) & 0xffffffff, np.asarray(b, np.int64) & 0xffffffff
                result = {'eq':np.equal, 'ne':np.not_equal, 'lt':np.less, 'le':np.less_equal,
                          'gt':np.greater, 'ge':np.greater_equal}[relation](a, b)
                assign(args[0], result, mask)
            elif opcode in ('selp.s32', 'selp.u32', 'selp.f32'):
                assign(args[0], np.where(value(args[3], mask), value(args[1], mask), value(args[2], mask)), mask)
            elif opcode in ('add.rn.f32', 'sub.rn.f32', 'mul.rn.f32', 'div.rn.f32',
                            'max.NaN.f32', 'min.NaN.f32', 'max.f32', 'min.f32'):
                a, b = np.asarray(value(args[1], mask), np.float32), np.asarray(value(args[2], mask), np.float32)
                fn = {'add.rn.f32':np.add, 'sub.rn.f32':np.subtract, 'mul.rn.f32':np.multiply,
                      'div.rn.f32':np.divide, 'max.NaN.f32':np.maximum, 'min.NaN.f32':np.minimum,
                      'max.f32':np.fmax, 'min.f32':np.fmin}[opcode]
                result = fn(a, b)
                if opcode.startswith(('max.', 'min.')):
                    negative = (np.signbit(a) & np.signbit(b)) if opcode.startswith('max.') else (np.signbit(a) | np.signbit(b))
                    zero = np.where(negative, np.float32(-0.0), np.float32(0.0))
                    result = np.where((a == 0) & (b == 0), zero, result)
                assign(args[0], np.asarray(result, np.float32), mask)
            elif opcode in ('sqrt.rn.f32', 'ex2.approx.f32', 'rcp.rn.f32'):
                a = np.asarray(value(args[1], mask), np.float32)
                fn = {'sqrt.rn.f32':np.sqrt, 'ex2.approx.f32':np.exp2, 'rcp.rn.f32':np.reciprocal}[opcode]
                assign(args[0], np.asarray(fn(a), np.float32), mask)
            elif re.fullmatch(r'(add|sub|mul\.lo|mul\.wide|mad\.lo|div|rem|min|max|and|shr)\.(u32|s32|u64|b32)', opcode):
                op, dtype = opcode.rsplit('.', 1)
                a, b = np.asarray(value(args[1], mask), np.int64), np.asarray(value(args[2], mask), np.int64)
                if dtype == 's32':
                    a, b = signed(a), signed(b)
                if op == 'add': result = a + b
                elif op == 'sub': result = a - b
                elif op.startswith('mul'): result = a * b
                elif op.startswith('mad'): result = a * b + value(args[3], mask)
                elif op in ('div', 'rem'):
                    if np.any(b == 0): raise AssertionError('modeled integer division by zero')
                    quotient = (np.abs(a) // np.abs(b)) * np.where((a < 0) != (b < 0), -1, 1)
                    result = quotient if op == 'div' else a - quotient * b
                elif op == 'min': result = np.minimum(a, b)
                elif op == 'max': result = np.maximum(a, b)
                elif op == 'and': result = a & b
                else: result = (a & 0xffffffff) >> b
                if dtype in ('u32', 's32', 'b32') and op != 'mul.wide':
                    result &= 0xffffffff
                assign(args[0], result, mask)
            else:
                raise AssertionError('unsupported modeled instruction: ' + opcode)
    if current_copies or groups or np.any(pending):
        raise AssertionError('kernel returns with unfinished async copies')
    for name, writes in counts.items():
        expected_atomic = name in launch.get('zero_outputs', [])
        if not np.all(writes >= 1 if expected_atomic else writes == 1):
            raise AssertionError('missing or overlapping output writes: ' + name)
    return MatmulResult({name: buffers[name] for name in outputs}, counts, writers, barrier_count, async_copy_bytes, committed_groups, completed_groups)
