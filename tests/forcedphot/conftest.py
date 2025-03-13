import pytest
import forcedphot

try:
    import lsst.geom
    import lsst.afw
except ImportError:
    def pytest_collection_modifyitems(config, items):
        print("Skipping LSST-dependent tests...")
        skip_lsst = pytest.mark.skip(reason="LSST modules not available")
        for item in items:
            item.add_marker(skip_lsst)
