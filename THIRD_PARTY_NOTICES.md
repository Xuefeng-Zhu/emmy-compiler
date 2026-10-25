# Third-party sources and notices

## Emmy dependency

`emmy-ml==0.3.11` is installed as a dependency, not vendored. Its distribution
metadata declares Apache-2.0 and includes LICENSE in the installed dist-info
licenses directory. Package licenses and notices remain intact. See
[Emmy](https://github.com/cloudrift-ai/emmy). Dependencies in the lock files retain
their own license metadata.

## Challenge reference materials

The local `.reference/` directory contains unmodified downloads from
[emmy-hackathon-2026](https://github.com/cloudrift-ai/emmy-hackathon-2026) at commit
`ec8cde1943f88efea3827a95bda6085553d2d407`. Paths and SHA-256 hashes are recorded
in [`docs/reference-manifest.json`](docs/reference-manifest.json). These JSON,
text, CUDA examples and documentation are used as fixtures and references;
`.reference/` is excluded from this repository's Git history.

The hash-pinned upstream `submit.py` client was used for the 15 authorized public
cases. Upstream generation scripts were not run, and reference implementation
code is not used as a lowering backend.

No LICENSE or NOTICE file was present in the pinned challenge repository tree.
Emmy's Apache-2.0 declaration does not establish a license for those separate
challenge materials. This project does not redistribute them or grant rights to
them.

## Original code

Original code in this public source repository is licensed under the
[Apache License, Version 2.0](LICENSE). Third-party dependencies and challenge
reference materials retain their separate terms; the project license does not
relicense them.

## Technical documentation

PTX syntax and semantics were checked against NVIDIA's
[Parallel Thread Execution reference](https://docs.nvidia.com/cuda/parallel-thread-execution/).
Assembly and execution evidence is recorded separately in
[`docs/RUNNER_RESULTS.md`](docs/RUNNER_RESULTS.md).
