def pytest_ignore_collect(collection_path, config):
    """Skip collecting test modules that require LSST."""

    if collection_path.name in ["test_image_service.py", "test_photometry_service.py", "test_odc.py"]:
        return True
    return False
