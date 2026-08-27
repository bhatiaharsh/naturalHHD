'''
This package now installs and imports as `pynhhd`, matching pynhhd-v1.1
(numpy) -- deliberately, so they're distinguished by version rather than
name (see pyproject.toml, and README.md's Status section). One
consequence: in an environment where this package is installed, plain
`import pynhhd` / `pytest.importorskip('pynhhd')` resolves to *this*
(torch) package, not the numpy one, even if pynhhd-v1.1 also happens to be
on sys.path. A few tests still want to cross-validate specific paths
against the numpy reference; `numpy_pynhhd` below loads it directly from
../../pynhhd-v1.1/pynhhd under a distinct module name so it doesn't
collide with (or get shadowed by) the installed `pynhhd`.
'''

import sys
import pathlib
import importlib.util

import pytest

_NUMPY_PYNHHD_MODNAME = 'pynhhd_numpy_reference'


def _load_numpy_pynhhd():
    root = pathlib.Path(__file__).resolve().parents[2] / 'pynhhd-v1.1' / 'pynhhd'
    init_path = root / '__init__.py'
    if not init_path.exists():
        return None

    if _NUMPY_PYNHHD_MODNAME in sys.modules:
        return sys.modules[_NUMPY_PYNHHD_MODNAME]

    spec = importlib.util.spec_from_file_location(
        _NUMPY_PYNHHD_MODNAME, init_path, submodule_search_locations=[str(root)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[_NUMPY_PYNHHD_MODNAME] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        del sys.modules[_NUMPY_PYNHHD_MODNAME]
        return None
    return module


@pytest.fixture
def numpy_pynhhd():
    '''pynhhd-v1.1 (numpy), loaded under a name distinct from the installed
    `pynhhd` (this torch package). Skips the test if it can't be loaded --
    e.g. pynhhd-v1.1 isn't present, or it hits one of its own known
    import-time issues (a missing `gsw`, or numpy>=1.24's removed
    np.float -- see docs/CODE_REVIEW_REPORT.md).'''
    module = _load_numpy_pynhhd()
    if module is None:
        pytest.skip('pynhhd-v1.1 (numpy) not importable; skipping cross-validation')
    return module
