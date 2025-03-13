def pytest_ignore_collect(path, config):
    """Skip collecting test modules that require LSST."""

    if path.basename in ['test_image_service.py', 'test_photometry_service.py', 'test_odc.py']:
        return True
    return False
