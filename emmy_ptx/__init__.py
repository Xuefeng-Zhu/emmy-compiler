from pathlib import Path

from .errors import CompileError


def compile_file(path: Path) -> str:
    """Compile a trusted Emmy IR and its adjacent meta.json entirely in memory."""
    from .loader import load_kernel
    from .ptx import emit

    return emit(load_kernel(Path(path)))


__all__ = ['compile_file', 'CompileError']
