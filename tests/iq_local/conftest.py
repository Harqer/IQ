import os

import pytest


def pytest_collection_modifyitems(config, items):
    net_on = os.environ.get("IQ_NET_TESTS") == "1"
    for item in items:
        if "net" in item.keywords and not net_on:
            item.add_marker(pytest.mark.skip(reason="IQ_NET_TESTS=1 required"))
