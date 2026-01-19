from __future__ import annotations

import sys
import importlib
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


def _purge(pkg: str) -> None:
    # usuń pkg i wszystko co pkg.*
    for name in list(sys.modules.keys()):
        if name == pkg or name.startswith(pkg + "."):
            del sys.modules[name]


@contextmanager
def use_method_src(src_dir: Path, pkg: str = "rekognition_online_action_detection") -> Iterator[None]:
    """
    Minimalny, stabilny scope:
    - dodaje src_dir na początek sys.path
    - czyści wskazany pakiet z sys.modules (żeby nie było konfliktów nazw)
    - przywraca stan po wyjściu
    """
    src_dir = src_dir.resolve()

    _purge(pkg)
    sys.path.insert(0, str(src_dir))
    importlib.invalidate_caches()

    try:
        yield
    finally:
        _purge(pkg)
        # usuń dokładnie ten wpis
        try:
            sys.path.remove(str(src_dir))
        except ValueError:
            pass
        importlib.invalidate_caches()
