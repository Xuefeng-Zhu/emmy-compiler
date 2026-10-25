"""Keep educational examples and embedded evidence tied to actual compiler behavior."""
import json
from pathlib import Path
import subprocess
import sys
import unittest
import numpy as np
from emmy_ptx import compile_file
from ptx_model import run_ptx

ROOT = Path(__file__).resolve().parents[1]


class LearningArtifactsTests(unittest.TestCase):
    def test_original_teaching_sample_compiles_and_covers_ten_outputs(self):
        folder = ROOT/'docs/explainer/sample'
        ptx = compile_file(folder/'learn_add.ir.json')
        self.assertEqual(ptx, (folder/'learn_add.ptx').read_text())
        launch, = json.loads((folder/'meta.json').read_text())['kernels']
        output = run_ptx(ptx, launch).outputs['result']
        np.testing.assert_array_equal(output, np.arange(10, dtype=np.float32)*3 + 1)
        self.assertEqual(launch['grid'][0]*launch['block'][0] - output.size, 6)

    def test_generated_page_and_provenance_are_reproducible(self):
        result = subprocess.run([sys.executable, '-B', str(ROOT/'tools/build_explainer.py'), '--check'],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_documentation_links_and_offline_contract(self):
        result = subprocess.run([sys.executable, '-B', str(ROOT/'tools/check_docs.py')],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
