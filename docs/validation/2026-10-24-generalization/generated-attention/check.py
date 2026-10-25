"""Generate fresh attention Kernel IR and check our compiler, entirely on CPU.

Run with the project's pinned Python from its repository root. Upstream Emmy
passes construct Kernel IR from a new Graph; no renderer or backend is called.
"""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--output', type=Path, default=Path('/tmp/emmy-generated-attention-check'))
    parser.add_argument('--strategy', choices=('emission-order', 'greedy', 'both'), default='both')
    options = parser.parse_args()
    repo = options.repo.resolve()
    sys.path[:0] = [str(repo), str(repo / 'tests')]
    # Scope scheduling evidence to shipped offline heuristics, avoiding a
    # machine-local model/database and the large measured golden corpus.
    os.environ['EMMY_GOLDEN_FILE'] = ''
    os.environ['EMMY_PRICE_BUDGET_S'] = '5'
    os.environ['EMMY_ONLINE_FILE'] = str(options.output / 'absent-online-prior.json')

    import numpy as np
    from emmy.compiler.context import Context
    from emmy.compiler.graph import Graph
    from emmy.compiler.ir.base import ConstantOp, InputOp
    from emmy.compiler.ir.expr import Literal
    from emmy.compiler.ir.frontend.ir import MatmulOp, SoftmaxOp
    from emmy.compiler.ir.kernel.ir import KernelOp
    from emmy.compiler.ir.tensor.ir import ElementwiseOp, IndexMapOp, IndexSource
    from emmy.compiler.pipeline import KERNEL_PASSES, Pipeline
    from emmy.compiler.pipeline.pipeline import Run
    from emmy.compiler.pipeline.search.policy.greedy import greedy_decide
    from emmy.compiler.tensor import Tensor
    from emmy_ptx import compile_file
    from ptx_model import input_buffers
    from structured_model import run_structured_ptx

    def attention_graph(rows, keys, depth, columns):
        graph = Graph()
        for name, shape in [('q', (rows, depth)), ('kt', (depth, keys)), ('v', (keys, columns))]:
            graph.add_node(InputOp(), [], Tensor(name, shape))
        graph.inputs = ['q', 'kt', 'v']
        scale = math.sqrt(depth)
        graph.add_node(ConstantOp(name='scale', value=scale), [],
                       Tensor('scale', (1,), constant=True, value=scale))
        graph.add_node(MatmulOp(), ['q', 'kt'], Tensor('scores', (rows, keys)))
        graph.add_node(IndexMapOp(out_shape=(rows, keys), sources=(IndexSource(0, (Literal(0, 'int'),)),)),
                       ['scale'], Tensor('scales', (rows, keys)))
        graph.add_node(ElementwiseOp('divide'), ['scores', 'scales'], Tensor('scaled', (rows, keys)))
        graph.add_node(SoftmaxOp(), ['scaled'], Tensor('weights', (rows, keys)))
        graph.add_node(MatmulOp(), ['weights', 'v'], Tensor('result', (rows, columns)))
        graph.outputs = ['result']
        return graph

    pipeline = Pipeline.build(KERNEL_PASSES)
    context = Context.from_target((12, 0), gpu_name='NVIDIA GeForce RTX 5090')
    strategies = ('emission-order', 'greedy') if options.strategy == 'both' else (options.strategy,)
    summary = {'emmy_version': importlib.metadata.version('emmy-ml'),
               'generation': 'new high-level Graph -> official KERNEL_PASSES',
               'gpu_execution': False, 'renderer_called': False,
               'scheduling': {'golden_evidence': 'disabled', 'online_evidence': 'absent path',
                              'structural_pricing_budget_s': 5}, 'cases': []}
    for strategy in strategies:
        destination = options.output if strategy == 'emission-order' else Path(str(options.output) + '-greedy')
        for shape in ((256, 256, 32, 32), (96, 200, 32, 72), (5, 9, 7, 11)):
            folder = destination / 'x'.join(map(str, shape))
            folder.mkdir(parents=True, exist_ok=True)
            source = attention_graph(*shape)
            if strategy == 'greedy':
                lowered = pipeline.run(source, ctx=context)
            else:
                lowered, _ = Run(pipeline, context).resolve(
                    source, greedy_decide(prior=None, price_structural=False))
            (folder / 'graph.json').write_text(json.dumps(lowered.to_dict(), indent=2, default=repr) + '\n')
            kernels = [node for node in lowered.nodes.values() if isinstance(node.op, KernelOp)]
            if len(kernels) != 1:
                raise AssertionError(f'expected one fused attention kernel, got {len(kernels)}')
            node, = kernels
            isolated = Graph()
            arguments = []
            for seed, name in enumerate(node.inputs):
                producer = lowered.nodes[name]
                tensor = producer.output
                constant = isinstance(producer.op, ConstantOp)
                isolated.add_node(producer.op if constant else InputOp(), [], tensor)
                descriptor = {'name': name, 'shape': [d.as_static() for d in tensor.shape],
                              'dtype': tensor.dtype.name}
                if constant:
                    descriptor['data'] = [producer.op.value]
                else:
                    descriptor['seed'] = seed
                    isolated.inputs.append(name)
                arguments.append(descriptor)
            isolated.add_node(node.op, node.inputs, outputs=node.outputs, node_id=node.id)
            isolated.outputs = [tensor.name for tensor in node.outputs]
            for tensor in node.outputs:
                arguments.append({'name': tensor.name, 'shape': [d.as_static() for d in tensor.shape] or [1],
                                  'dtype': tensor.dtype.name, 'output': True})
            tile, = node.op.body
            threads = tile.block_threads or 256
            meta = {'rtol': 1e-3, 'atol': 1e-4, 'kernels': [{
                'node': node.id, 'entry': node.op.name, 'arg_order': [arg['name'] for arg in arguments],
                'args': arguments, 'grid': [(tile.n_elements + threads - 1) // threads, 1, 1],
                'block': [threads, 1, 1], 'smem_bytes': node.op.smem_bytes(), 'zero_outputs': []}]}
            ir_path = folder / (node.id + '.ir.json')
            ir_path.write_text(json.dumps(isolated.to_dict(), indent=2, default=repr) + '\n')
            (folder / 'meta.json').write_text(json.dumps(meta, indent=2) + '\n')
            ptx = compile_file(ir_path)
            ir_path.with_suffix('.ptx').write_text(ptx)
            record = {'shape_rows_keys_depth_columns': list(shape), 'strategy': strategy,
                      'node_types': dict(Counter(type(n.op).__name__ for n in lowered.nodes.values())),
                      'statement_types': dict(Counter(type(statement).__name__ for statement in node.op)),
                      'compile': 'PASS', 'ir_sha256': hashlib.sha256(ir_path.read_bytes()).hexdigest(), 'ptx_sha256': hashlib.sha256(ptx.encode()).hexdigest(),
                      'ir_path': str(ir_path), 'metadata_path': str(folder / 'meta.json')}
            condition_ops = Counter()
            def count_condition_ops(expression):
                operation = getattr(expression, 'op', None)
                if operation is not None:
                    condition_ops[operation] += 1
                for field in ('left', 'right'):
                    child = getattr(expression, field, None)
                    if child is not None:
                        count_condition_ops(child)
            for statement in node.op:
                if type(statement).__name__ == 'Cond':
                    count_condition_ops(statement.cond)
            record['condition_operators'] = dict(condition_ops)
            if shape == (5, 9, 7, 11):
                launch = meta['kernels'][0]
                inputs = input_buffers(launch)
                scores = inputs['q'].astype(np.float64) @ inputs['kt'].astype(np.float64) / float(inputs['scale'][0])
                weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
                weights /= weights.sum(axis=-1, keepdims=True)
                expected = (weights @ inputs['v'].astype(np.float64)).astype(np.float32)
                actual = run_structured_ptx(ptx, launch)
                np.testing.assert_allclose(actual.outputs['result'], expected, rtol=meta['rtol'], atol=meta['atol'])
                record['cpu_operand_model'] = {'result': 'PASS',
                    'max_abs_error': float(np.max(np.abs(actual.outputs['result'] - expected))),
                    'stores_once': bool(np.all(actual.store_counts['result'] == 1))}
            summary['cases'].append(record)
            print(json.dumps(record), flush=True)
    summary_path = options.output.with_name(options.output.name + '-summary.json')
    summary_path.write_text(json.dumps(summary, indent=2) + '\n')
    print('SUMMARY', summary_path)


if __name__ == '__main__':
    main()
