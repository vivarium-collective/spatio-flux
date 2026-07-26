import os
import glob
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---- Task 1: workspace manifest + build_core -------------------------------

def test_workspace_yaml_present_and_shaped():
    with open(os.path.join(REPO, "workspace.yaml")) as f:
        ws = yaml.safe_load(f)
    assert ws["schema_version"] == 2
    assert ws["name"] == "spatio_flux"
    assert ws["package_path"] == "spatio_flux"
    assert ws["runtime"]["default_emitter"] == "sqlite"


def test_build_core_constructs():
    from spatio_flux.core import build_core
    core = build_core()
    # custom spatio-flux types registered
    assert core.access("concentration") is not None
    assert core.access("particle") is not None
