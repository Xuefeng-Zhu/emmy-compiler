"""CPU-only tests of the official IR interface and our PTX text emitter.

These assertions do not assemble PTX and do not execute a CUDA kernel.
"""

import copy
import json
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from emmy.compiler.graph import Graph

from emmy_ptx import compile_file
from emmy_ptx.errors import CompileError


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / ".reference" / "problems"


def official_ir(problem="p00_scalar_add"):
    paths = sorted((REFERENCE / problem).glob("*.ir.json"))
    if len(paths) != 1:
        raise AssertionError(f"Expected one official IR fixture in {REFERENCE / problem}")
    return paths[0]


class CompilerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory(prefix="emmy-test-")
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.path = self.directory / "kernel.ir.json"
        self.meta_path = self.directory / "meta.json"
        self.reset_fixture()

    def reset_fixture(self, problem="p00_scalar_add"):
        source = official_ir(problem)
        self.ir = json.loads(source.read_text())
        self.meta = json.loads((source.parent / "meta.json").read_text())
        self.save()

    def save(self):
        self.path.write_text(json.dumps(self.ir))
        self.meta_path.write_text(json.dumps(self.meta))

    def kernel(self):
        return next(node for node in self.ir["nodes"].values() if node["op"] == "KernelOp")

    def replace_body(self, old, new):
        body = self.kernel()["op_fields"]["body"]
        self.assertIn(old, body[0], "Fixture changed; update the mutation deliberately")
        body[0] = body[0].replace(old, new)
        self.save()

    def run_cli(self, *args):
        return subprocess.run(
            [sys.executable, str(ROOT / "compile.py"), *map(str, args)],
            cwd=self.directory,
            capture_output=True,
            text=True,
            timeout=30,
        )

    def test_official_json_deserializes_to_typed_statements(self):
        graph = Graph.from_dict(copy.deepcopy(self.ir))
        kernel = graph.nodes["add"].op
        self.assertEqual(type(kernel).__name__, "KernelOp")
        self.assertEqual(kernel.name, "k_add_pointwise")
        tile = kernel.body[0]
        self.assertEqual(type(tile).__name__, "Tile")
        self.assertEqual(tile.axes[0].name, "a0")
        self.assertEqual(
            [type(stmt).__name__ for stmt in tile.body],
            ["Load", "Load", "Assign", "Load", "Load", "Assign", "Write"],
        )
        self.assertEqual(len(tile.body[-1].values), 2)

    def test_emits_ptx_header_entry_and_pointer_signature(self):
        ptx = compile_file(self.path)
        self.assertTrue(ptx.startswith(".version "))
        self.assertIn(".target sm_89", ptx)
        self.assertIn(".address_size 64", ptx)
        self.assertRegex(ptx, r"\.entry\s+k_add_pointwise\s*\(")
        self.assertEqual(len(re.findall(r"\.param\s+\.u64\b", ptx)), 3)
        ptx.encode("ascii")

    def test_emits_global_thread_index_and_tail_guard(self):
        ptx = compile_file(self.path)
        for builtin in ("%ctaid.x", "%ntid.x", "%tid.x"):
            self.assertIn(builtin, ptx)
        self.assertRegex(ptx, r"setp\.[^;]*\b64\b[^;]*;")
        self.assertRegex(ptx, r"@!?%[^\s]+\s+(?:bra|ret)\b")

    def test_lowers_two_independent_adds_and_two_scalar_stores(self):
        ptx = compile_file(self.path)
        self.assertEqual(len(re.findall(r"\badd(?:\.rn)?\.f32\b", ptx)), 2)
        self.assertEqual(len(re.findall(r"\bld\.global\.f32\b", ptx)), 4)
        self.assertEqual(len(re.findall(r"\bst\.global\.f32\b", ptx)), 2)
        self.assertIn("cvta.to.global.u64", ptx)

    def test_compilation_is_deterministic(self):
        self.assertEqual(compile_file(self.path), compile_file(self.path))

    def test_vector_problem_uses_same_lowering_with_512_cell_guard(self):
        self.reset_fixture("p01_vector_add")
        ptx = compile_file(self.path)
        self.assertRegex(ptx, r"setp\.[^;]*\b512\b[^;]*;")
        self.assertEqual(len(re.findall(r"\badd(?:\.rn)?\.f32\b", ptx)), 2)

    def test_metadata_argument_order_controls_signature(self):
        self.meta["kernels"][0]["arg_order"] = ["add", "x1", "x0"]
        self.save()
        ptx = compile_file(self.path)
        parameters = re.findall(r"\.param\s+\.u64\s+([A-Za-z_][A-Za-z_0-9]*)", ptx)
        self.assertEqual(len(parameters), 3)
        for parameter, original_name in zip(parameters, ("add", "x1", "x0")):
            self.assertIn(original_name, parameter)

    def test_entry_name_comes_from_ir(self):
        self.kernel()["op_fields"]["name"] = "custom_pointwise_entry"
        self.meta["kernels"][0]["entry"] = "custom_pointwise_entry"
        self.save()
        ptx = compile_file(self.path)
        self.assertRegex(ptx, r"\.entry\s+custom_pointwise_entry\s*\(")
        self.assertNotIn(".entry k_add_pointwise", ptx)

    def test_changed_small_shape_is_not_hardcoded(self):
        for node in self.ir["nodes"].values():
            for output in node["outputs"]:
                output["shape"] = [6]
        for arg in self.meta["kernels"][0]["args"]:
            arg["shape"] = [6]
        self.replace_body("Dim(64)", "Dim(3)")
        ptx = compile_file(self.path)
        self.assertRegex(ptx, r"setp\.[^;]*\b3\b[^;]*;")

    def test_missing_input_file_is_a_compile_error(self):
        with self.assertRaises(CompileError):
            compile_file(self.directory / "missing.ir.json")

    def test_invalid_json_is_a_compile_error(self):
        self.path.write_text("{broken")
        with self.assertRaises(CompileError):
            compile_file(self.path)

    def test_missing_metadata_is_a_compile_error(self):
        self.meta_path.unlink()
        with self.assertRaises(CompileError):
            compile_file(self.path)

    def test_invalid_metadata_is_a_compile_error(self):
        self.meta_path.write_text("{broken")
        with self.assertRaises(CompileError):
            compile_file(self.path)

    def test_unknown_arithmetic_fails_clearly(self):
        self.replace_body("ElementwiseImpl('add')", "ElementwiseImpl('mul')")
        with self.assertRaisesRegex(CompileError, r"(?i)(mul|assign|unsupported)"):
            compile_file(self.path)

    def test_undefined_ssa_value_is_rejected(self):
        self.replace_body("args=('in0__u0', 'in1__u0')", "args=('missing_value', 'in1__u0')")
        with self.assertRaises(CompileError):
            compile_file(self.path)

    def test_atomic_write_is_rejected(self):
        self.replace_body("atomic=False", "atomic=True")
        with self.assertRaises(CompileError):
            compile_file(self.path)

    def test_unsupported_index_operator_is_rejected(self):
        self.replace_body("BinaryExpr(op='*'", "BinaryExpr(op='/'")
        with self.assertRaises(CompileError):
            compile_file(self.path)

    def test_unknown_statement_is_rejected(self):
        self.kernel()["op_fields"]["body"] = ["Sync()"]
        self.save()
        with self.assertRaises(CompileError):
            compile_file(self.path)

    def test_non_f32_buffer_is_rejected(self):
        self.ir["nodes"]["x0"]["outputs"][0]["dtype"] = "f64"
        self.meta["kernels"][0]["args"][0]["dtype"] = "f64"
        self.save()
        with self.assertRaises(CompileError):
            compile_file(self.path)

    def test_mismatched_entry_is_rejected(self):
        self.meta["kernels"][0]["entry"] = "wrong_entry"
        self.save()
        with self.assertRaises(CompileError):
            compile_file(self.path)

    def test_duplicate_argument_is_rejected(self):
        self.meta["kernels"][0]["arg_order"] = ["x0", "x0", "add"]
        self.save()
        with self.assertRaises(CompileError):
            compile_file(self.path)

    def test_unknown_argument_is_rejected(self):
        self.meta["kernels"][0]["arg_order"] = ["x0", "unknown", "add"]
        self.save()
        with self.assertRaises(CompileError):
            compile_file(self.path)

    def test_multidimensional_launch_is_rejected(self):
        self.meta["kernels"][0]["block"] = [128, 2, 1]
        self.save()
        with self.assertRaises(CompileError):
            compile_file(self.path)

    def test_insufficient_launch_coverage_is_rejected(self):
        self.reset_fixture("p01_vector_add")
        self.meta["kernels"][0]["grid"] = [1, 1, 1]
        self.save()
        with self.assertRaises(CompileError):
            compile_file(self.path)

    def test_cli_requires_exactly_one_positional_input(self):
        for args in ((), (self.path, self.path)):
            with self.subTest(args=args):
                result = self.run_cli(*args)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, "")
                self.assertTrue(result.stderr.strip())

    def test_cli_failure_has_no_partial_ptx_stdout(self):
        self.replace_body("ElementwiseImpl('add')", "ElementwiseImpl('mul')")
        result = self.run_cli(self.path)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertTrue(result.stderr.strip())
        self.assertNotIn("Traceback", result.stderr)

    def test_cli_success_from_another_working_directory(self):
        result = self.run_cli(self.path)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        self.assertEqual(result.stdout, compile_file(self.path))

    def test_out_of_bounds_memory_access_is_rejected(self):
        self.replace_body("Literal(value=0, dtype='int')", "Literal(value=128, dtype='int')")
        with self.assertRaisesRegex(CompileError, r"(?i)out.of.bounds"):
            compile_file(self.path)

    def test_duplicate_ssa_definition_is_rejected(self):
        self.replace_body("Assign(name='v0__u0'", "Assign(name='in0__u0'")
        with self.assertRaisesRegex(CompileError, r"(?i)duplicate"):
            compile_file(self.path)

    def test_negative_index_is_rejected(self):
        self.replace_body("Literal(value=0, dtype='int')", "Literal(value=-1, dtype='int')")
        with self.assertRaises(CompileError):
            compile_file(self.path)

    def test_unbound_index_variable_is_rejected(self):
        self.replace_body("Var(name='a0')", "Var(name='unbound_axis')")
        with self.assertRaises(CompileError):
            compile_file(self.path)

    def test_zero_tile_extent_is_rejected(self):
        self.replace_body("Dim(64)", "Dim(0)")
        with self.assertRaises(CompileError):
            compile_file(self.path)

    def test_nonlinear_index_is_rejected(self):
        self.replace_body("right=Literal(value=2, dtype='int')", "right=Var(name='a0')")
        with self.assertRaises(CompileError):
            compile_file(self.path)

    def test_overlapping_per_thread_writes_are_rejected(self):
        body = self.kernel()["op_fields"]["body"][0]
        start = body.index("Write(output='add', index=(")
        original_index = (
            "BinaryExpr(op='+', left=BinaryExpr(op='*', left=Var(name='a0'), "
            "right=Literal(value=2, dtype='int')), right=Literal(value=0, dtype='int'))"
        )
        self.assertIn(original_index, body[start:])
        self.kernel()["op_fields"]["body"][0] = (
            body[:start] + body[start:].replace(original_index, "Var(name='a0')", 1)
        )
        self.save()
        with self.assertRaises(CompileError):
            compile_file(self.path)

    def test_incomplete_output_coverage_is_rejected(self):
        self.replace_body("values=('v0__u0', 'v0__u1')", "values=('v0__u0',)")
        with self.assertRaises(CompileError):
            compile_file(self.path)

    def test_oversized_index_arithmetic_is_rejected(self):
        self.replace_body("Literal(value=2, dtype='int')", "Literal(value=1073741824, dtype='int')")
        with self.assertRaises(CompileError):
            compile_file(self.path)

    def test_unhashable_metadata_argument_name_is_a_compile_error(self):
        self.meta["kernels"][0]["args"][0]["name"] = []
        self.save()
        with self.assertRaises(CompileError):
            compile_file(self.path)

    def test_malformed_metadata_cli_has_no_traceback(self):
        self.meta["kernels"][0]["args"][0]["name"] = []
        self.save()
        result = self.run_cli(self.path)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertTrue(result.stderr.strip())
        self.assertNotIn("Traceback", result.stderr)

    def test_missing_statement_dtype_is_a_compile_error(self):
        body = self.kernel()["op_fields"]["body"]
        body[0] = body[0].replace(
            "dtype=DataType(name='f32', np=dtype('float32'), nbytes=4)",
            "dtype=None",
            1,
        )
        self.save()
        with self.assertRaises(CompileError):
            compile_file(self.path)


if __name__ == "__main__":
    unittest.main()
