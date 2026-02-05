from __future__ import annotations

import sys
import importlib
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Iterable


def _match(name: str, pkg: str) -> bool:
    return name == pkg or name.startswith(pkg + ".")


def _collect_modules(pkgs: Iterable[str]) -> dict[str, object]:
    pkgs = tuple([p for p in pkgs if p])
    saved: dict[str, object] = {}
    for name, mod in list(sys.modules.items()):
        if any(_match(name, p) for p in pkgs):
            saved[name] = mod
    return saved


def _purge_many(pkgs: Iterable[str]) -> None:
    pkgs = tuple([p for p in pkgs if p])
    for name in list(sys.modules.keys()):
        if any(_match(name, p) for p in pkgs):
            del sys.modules[name]


@contextmanager
def use_method_src(
    src_dir: Path,
    pkg: str | None = "rekognition_online_action_detection",
    purge_extra: Iterable[str] = (),
    restore_extra: bool = True,
) -> Iterator[None]:
    """
    - dodaje src_dir na początek sys.path
    - purguje `pkg` (jak dotychczas) + opcjonalnie `purge_extra`
    - (opcjonalnie) przywraca moduły z `purge_extra` po wyjściu, żeby nie kasować Twoich globalnych importów
    """
    src_dir = src_dir.resolve()
    main_pkg = (pkg,) if pkg else tuple()
    extra_pkgs = tuple(purge_extra)

    saved_extra = _collect_modules(extra_pkgs) if restore_extra and extra_pkgs else {}

    _purge_many(main_pkg)
    _purge_many(extra_pkgs)

    sys.path.insert(0, str(src_dir))
    importlib.invalidate_caches()

    try:
        yield
    finally:
        _purge_many(main_pkg)
        _purge_many(extra_pkgs)

        try:
            sys.path.remove(str(src_dir))
        except ValueError:
            pass

        importlib.invalidate_caches()

        if saved_extra:
            sys.modules.update(saved_extra)
