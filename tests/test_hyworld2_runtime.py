from modal_world.hyworld2_runtime import HYWORLD2_ARTIFACT_BUNDLES


def test_runtime_bundles_are_content_addressed():
    assert len(HYWORLD2_ARTIFACT_BUNDLES) == 4
    for spec in HYWORLD2_ARTIFACT_BUNDLES:
        assert len(spec["archive_sha256"]) == 64
        int(spec["archive_sha256"], 16)


def test_hy_derived_bundle_is_not_public():
    by_tag = {spec["tag"]: spec for spec in HYWORLD2_ARTIFACT_BUNDLES}
    assert by_tag["hyworld2-hy-native-py311-cu128-torch271-sm120-v1"]["public_release"] is False
