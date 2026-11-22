import os

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "net: touches the network; requires IQ_NET_TESTS=1"
    )
    import warnings
    warnings.filterwarnings(
        "ignore",
        message=".*hf_xet.download_files\\(\\) is deprecated.*",
        category=DeprecationWarning,
    )


def pytest_collection_modifyitems(config, items):
    net_on = os.environ.get("IQ_NET_TESTS") == "1"
    for item in items:
        if "net" in item.keywords and not net_on:
            item.add_marker(pytest.mark.skip(reason="IQ_NET_TESTS=1 required"))
    deprecation_filter = pytest.mark.filterwarnings(
        "ignore:hf_xet.download_files:DeprecationWarning"
    )
    for item in items:
        item.add_marker(deprecation_filter)
