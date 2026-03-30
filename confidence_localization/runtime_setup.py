import logging
import sys
import warnings
from pathlib import Path


def configure_runtime(entry_file: str) -> Path:
    warnings.filterwarnings("ignore", message=r"pkg_resources is deprecated as an API.*")
    warnings.filterwarnings("ignore", message=r"Deprecated call to `pkg_resources\.declare_namespace.*")
    logging.getLogger("torch.distributed.nn.jit.instantiator").setLevel(logging.WARNING)

    repo_root = Path(entry_file).resolve().parents[1]
    for rel_path in ("confidence_localization", "data"):
        abs_path = repo_root / rel_path
        if str(abs_path) not in sys.path:
            sys.path.append(str(abs_path))
    return repo_root
