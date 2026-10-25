# Third-party sources and notices

`emmy-ml==0.3.11` is installed as a dependency, not vendored. Its distribution
metadata declares Apache-2.0 and includes LICENSE in the installed dist-info
licenses directory. That file and all package notices remain intact. See
[Emmy](https://github.com/cloudrift-ai/emmy). The dependency lock also includes its
transitive dependencies; each retains its own installed license metadata.

Local `.reference/` files are unmodified downloads from
https://github.com/cloudrift-ai/emmy-hackathon-2026 at commit
`ec8cde1943f88efea3827a95bda6085553d2d407`. Their exact paths and SHA-256 hashes are
in `docs/reference-manifest.json`. These include official JSON/text/CUDA examples
and documentation used for development and comparison. No upstream script is
executed. No reference implementation code is used as a lowering backend.

No LICENSE or NOTICE file was present in that challenge repository tree when
checked. Do not assume Emmy's Apache-2.0 license automatically covers the challenge
materials. They are excluded from this project's Git history, and permission to
redistribute them must be confirmed before publication. This notice grants no
rights in third-party material.

Our original code has no public license selected yet. This initial local repository
is not a published entry. Before publishing, select a license for original work,
check dependency notice obligations, confirm challenge-material rights and review
competition entry/IP terms with the user. No legal agreement was accepted here.

PTX syntax was checked against NVIDIA's primary reference:
https://docs.nvidia.com/cuda/parallel-thread-execution/ . That review does not
substitute for assembly or hardware execution.
