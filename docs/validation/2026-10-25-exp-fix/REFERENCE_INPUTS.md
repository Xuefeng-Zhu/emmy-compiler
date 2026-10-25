# Official expected arrays

The three official expected.npz files for f02_row_softmax, a01_attention and
a02_attention_128 are intentionally excluded from this archive. Prepare them
with tools/fetch_reference.py under the ignored .reference/problems/ directory.
Their exact SHA256 values remain in each report.json input_artifact_sha256 map
and the immutable reference manifest.

Generated PTX, complete returned GPU output arrays, comparison outcomes and
stdout/stderr are retained. Synthetic expected arrays were calculated by this
project's own recorded NumPy/math oracles. Removing the upstream archive copies
does not change any compiler source, PTX, expected value, threshold or test result.
