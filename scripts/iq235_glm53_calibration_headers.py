"""Read-only audit of archived 6144->4096 GLM coordinate-map headers."""
from __future__ import annotations
import json

from iq_cloud.glm53_calibration_headers import inspect_remote_coordinate_maps


if __name__ == "__main__":
    print(json.dumps(inspect_remote_coordinate_maps(), indent=2, sort_keys=True))
