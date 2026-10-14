"""Real MSC/Megatron command contract tests, without model-weight placeholders."""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from iq_transfer.megatron_bridge import (
    DONOR_PINS, GLM53_ID, GLM53_REVISION, NEMOTRON_ID,
    MegatronBridgeIntegrationError, make_import_plan, render_plan,
)
from iq_transfer.megatron_storage import (
    MegatronStorageError, StorageProfile, mocha_profile,
    load_profile_spec, render_msc_config, msc_uri,
)


class MSCStorageContractTests(unittest.TestCase):
    def test_mocha_s3_gateway_has_actual_hf_endpoint_and_required_botocore_options(self):
        data = json.loads(render_msc_config(
            [mocha_profile()], cache_dir="/mnt/nvme/msc-cache"
        ))
        entry = data["profiles"]["mocha"]
        self.assertEqual(entry["storage_provider"]["type"], "s3")
        opts = entry["storage_provider"]["options"]
        self.assertEqual(opts["base_path"], "mocha")
        self.assertEqual(opts["endpoint_url"], "https://s3.hf.co/kofecheckmate")
        self.assertEqual(opts["s3"], {"addressing_style": "path"})
        self.assertEqual(opts["request_checksum_calculation"], "when_required")
        self.assertEqual(opts["response_checksum_validation"], "when_required")
        self.assertEqual(entry["credentials_provider"]["options"], {
            "access_key": "${HF_S3_ACCESS_KEY}",
            "secret_key": "${HF_S3_SECRET_KEY}",
        })
        self.assertNotIn("HFAK", json.dumps(data))

    def test_cloud_profiles_are_separate_and_compatible(self):
        profiles = [
            mocha_profile(),
            StorageProfile("training", "s3", "iq-train", region_name="us-west-2"),
            StorageProfile("data", "gcs", "iq-data", project_id="research-prod"),
            StorageProfile("artifacts", "azure", "iq-artifacts",
                           endpoint_url="https://trainingaccount.blob.core.windows.net"),
            StorageProfile("oracle", "oci", "iq-models", namespace="iq-namespace"),
        ]
        result = json.loads(render_msc_config(
            profiles, cache_dir="/scratch/msc", cache_size="100G"
        ))
        self.assertEqual(len(result["profiles"]), 5)
        self.assertEqual(result["profiles"]["data"]["storage_provider"]["type"], "gcs")
        self.assertEqual(result["profiles"]["artifacts"]["storage_provider"]["type"], "azure")
        self.assertEqual(result["profiles"]["oracle"]["storage_provider"]["options"]["namespace"], "iq-namespace")
        self.assertEqual(msc_uri("training", "checkpoints/iter_0000000"), "msc://training/checkpoints/iter_0000000")

    def test_user_supplied_profile_spec_preserves_backend_options_without_tokens(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "profiles.json"
            path.write_text(json.dumps({"profiles": [
                {"name": "other", "provider": "s3",
                 "bucket": "model-bucket", "endpoint_url": "https://storage.example.org",
                 "region_name": "us-east-1", "access_key_env": "OTHER_ACCESS",
                 "secret_key_env": "OTHER_SECRET"}
            ]}), encoding="utf-8")
            loaded = load_profile_spec(path)
            data = json.loads(render_msc_config(
                [mocha_profile(), *loaded], cache_dir="/scratch/msc"
            ))
            self.assertEqual(data["profiles"]["other"]["credentials_provider"]["options"]["access_key"], "${OTHER_ACCESS}")
            self.assertEqual(data["profiles"]["other"]["storage_provider"]["options"]["endpoint_url"], "https://storage.example.org")

    def test_invalid_profile_and_directory_traversal_fail_closed(self):
        with self.assertRaises(MegatronStorageError):
            render_msc_config([mocha_profile(), mocha_profile()], cache_dir="/scratch/msc")
        with self.assertRaises(MegatronStorageError):
            StorageProfile("_reserved", "s3", "bucket")
        with self.assertRaises(MegatronStorageError):
            StorageProfile("other", "s3", "bucket", access_key_env="MY_ACTUAL_TOKEN")
        with self.assertRaises(MegatronStorageError):
            StorageProfile("other", "azure", "mycontainer")
        with self.assertRaises(MegatronStorageError):
            msc_uri("mocha", "../hidden")
        with self.assertRaises(MegatronStorageError):
            render_msc_config([mocha_profile()], cache_dir="relative/cache")


class MegatronImportCommandTests(unittest.TestCase):
    def kwargs(self, output: str) -> dict:
        return dict(source=NEMOTRON_ID, donor="nemotron-super",
                    output_dir=output, bridge_root="/opt/Megatron-Bridge",
                    executor="local", device="gpu", gpus_per_node=8,
                    tensor_parallel=1, pipeline_parallel=1,
                    expert_parallel=8, expert_tensor_parallel=1)

    def test_official_nemotron_super_conversion_command_is_pinned(self):
        with TemporaryDirectory() as temp:
            plan = make_import_plan(**self.kwargs(temp + "/converted"))
        command = list(plan.command)
        self.assertEqual(command[:4], [
            "/opt/Megatron-Bridge/scripts/conversion/convert.sh",
            "import", "--executor", "local"
        ])
        self.assertEqual(command[command.index("--hf-revision") + 1],
                         DONOR_PINS["nemotron-super"][1])
        self.assertEqual(command[command.index("--ep") + 1], "8")
        self.assertEqual(plan.iq_target_blocks, 78)
        self.assertFalse(plan.native_donor_to_iq_equivalence)
        self.assertTrue(plan.requires_iq_operator_mapping)
        self.assertEqual(json.loads(render_plan(plan))["official_repo"], NEMOTRON_ID)

    def test_glm53_uses_official_glm5_bridge_revision(self):
        with TemporaryDirectory() as temp:
            plan = make_import_plan(
                donor="glm53", source=GLM53_ID,
                output_dir=temp + "/converted",
                bridge_root="/opt/Megatron-Bridge",
                device="gpu", gpus_per_node=8, expert_parallel=8
            )
        self.assertEqual(plan.official_revision, GLM53_REVISION)
        self.assertIn("--torch-dtype", plan.command)
        self.assertEqual(plan.command[plan.command.index("--torch-dtype") + 1], "bfloat16")

    def test_mounted_mocha_checkpoint_is_accepted_but_not_rewritten(self):
        with TemporaryDirectory() as temp:
            source = Path(temp) / "mounted" / "nvidia"
            source.mkdir(parents=True)
            (source / "config.json").write_text('{"model_type":"nemotron_h"}')
            output = Path(temp) / "conversion"
            plan = make_import_plan(
                **{**self.kwargs(str(output)), "source": str(source)}
            )
            self.assertIn(str(source), plan.command)
            self.assertNotIn("--hf-revision", plan.command)
            self.assertFalse(output.exists())

    def test_direct_object_store_source_and_existing_output_are_rejected(self):
        with TemporaryDirectory() as temp:
            args = self.kwargs(temp + "/conversion")
            with self.assertRaisesRegex(MegatronBridgeIntegrationError, "mounted/local"):
                make_import_plan(**{**args, "source": "msc://mocha/nvidia"})
            output = Path(temp) / "conversion"
            output.mkdir()
            (output / "iter_0000000").mkdir()
            with self.assertRaisesRegex(MegatronBridgeIntegrationError, "nonempty"):
                make_import_plan(**args)

    def test_device_and_parallelism_are_explicit(self):
        with TemporaryDirectory() as temp:
            args = self.kwargs(temp + "/converted")
            with self.assertRaisesRegex(MegatronBridgeIntegrationError, "requires explicit"):
                make_import_plan(**{**args, "gpus_per_node": 0})
            with self.assertRaisesRegex(MegatronBridgeIntegrationError, "positive"):
                make_import_plan(**{**args, "tensor_parallel": 0})


if __name__ == "__main__":
    unittest.main()
