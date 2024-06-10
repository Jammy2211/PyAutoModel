import pytest
from autoconf.dictable import to_dict, from_dict

import autogalaxy as ag
from autogalaxy.profiles.geometry_profiles import GeometryProfile


@pytest.fixture(name="ell_sersic")
def make_ell_sersic():
    return ag.mp.Sersic()


@pytest.fixture(name="ell_sersic_dict")
def make_ell_sersic_dict():
    return {
        "type": "instance",
        "class_path": "autogalaxy.profiles.mass.stellar.sersic.Sersic",
        "arguments": {
            "mass_to_light_ratio": 1.0,
            "ell_comps": {"type": "tuple", "values": [0.0, 0.0]},
            "sersic_index": 0.6,
            "intensity": 0.1,
            "effective_radius": 0.6,
            "centre": {"type": "tuple", "values": [0.0, 0.0]},
        },
    }


def test_to_dict(ell_sersic, ell_sersic_dict):
    assert to_dict(ell_sersic) == ell_sersic_dict


def test_from_dict(ell_sersic, ell_sersic_dict):
    assert ell_sersic == from_dict(ell_sersic_dict)
