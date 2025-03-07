import pytest

try:
    import lsst.geom
    import lsst.afw
except ImportError:
    pytest.skip("LSST modules not available", allow_module_level=True)