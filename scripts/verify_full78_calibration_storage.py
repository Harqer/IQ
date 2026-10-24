"""Independent bucket readback gate: verify all 78 WARM layer artifacts exist."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

MAP_KEYS = ("residual_map", "attention_input_map", "moe_input_map",
            "compressed_kv_map", "latent_map", "dense_intermediate_map")


def verify(root: Path) -> dict:
    manifest = root / "glm53_calibration.json"
    if not manifest.is_file():
        raise RuntimeError("FULL78_CALIBRATION_MISSING: no manifest")
    metadata=json.loads(manifest.read_text(encoding="utf-8"))
    records=metadata.get("stages",[])
    if (len(records)!=78 or metadata.get("source_layers")!=78 or
            [stage.get("source_layer") for stage in records]!=list(range(78))):
        raise RuntimeError("FULL78_CALIBRATION_MANIFEST_INVALID")
    if sum(isinstance(s.get("expert_usage"),str) for s in records)!=75:
        raise RuntimeError("FULL78_CALIBRATION_ROUTER_COUNT_INVALID")
    expected={"glm53_calibration.json"}
    for stage in records:
        for key in MAP_KEYS:
            name=stage.get(key)
            if name is None:
                continue
            if not isinstance(name,str) or "/" in name or ".." in name:
                raise RuntimeError("unsafe calibration file")
            expected.add(name+".json")
            expected.add(name+".safetensors")
        if stage.get("expert_usage"):
            expected.add(stage["expert_usage"])
    for key in ("final_output_map","lexical_input_map"):
        name=metadata.get(key)
        if not isinstance(name,str):
            raise RuntimeError(f"missing {key}")
        expected.add(name+".json")
        expected.add(name+".safetensors")
    missing=[name for name in sorted(expected) if not (root/name).is_file()
             or (root/name).stat().st_size<=0]
    if missing:
        raise RuntimeError(
            f"FULL78_CALIBRATION_MISSING_FILES={len(missing)} "
            f"first_missing={missing[:15]}"
        )
    if len(expected)<800:
        raise RuntimeError("unexpectedly small 78-stage calibration file inventory")
    return {"full78_calibration":"verified_files_present",
            "expected_artifacts":len(expected),
            "dense_layers":3, "routed_layers":75}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration",type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(verify(args.calibration),sort_keys=True),flush=True)


if __name__=="__main__":
    main()
