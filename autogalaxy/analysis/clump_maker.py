from typing import Type, Optional

import autofit as af
import autoarray as aa

from autogalaxy.galaxy.galaxy import Galaxy
from autogalaxy.profiles.light_profiles.light_profiles import LightProfile
from autogalaxy.profiles.mass_profiles.mass_profiles import MassProfile


class ClumpModel:
    def __init__(
        self,
        redshift: float,
        centres: aa.Grid2DIrregular,
        light_cls: Optional[Type[LightProfile]] = None,
        mass_cls: Optional[Type[MassProfile]] = None,
    ):

        self.redshift = redshift
        self.centres = centres

        self.light_cls = light_cls
        self.mass_cls = mass_cls

    @property
    def total_clumps(self):
        return len(self.centres.in_list)

    @property
    def light_list(self):
        if self.light_cls is not None:
            return [
                af.Model(self.light_cls, centre=centre)
                for centre in self.centres.in_list
            ]

    @property
    def mass_list(self):

        if self.mass_cls is not None:
            return [
                af.Model(self.mass_cls, centre=centre)
                for centre in self.centres.in_list
            ]

    @property
    def clump_dict(self):

        clump_dict = {}

        for i in range(self.total_clumps):

            light = self.light_list[i] if self.light_cls is not None else None
            mass = self.mass_list[i] if self.mass_cls is not None else None

            clump_dict[f"clump_{i}"] = af.Model(
                Galaxy, redshift=self.redshift, light=light, mass=mass
            )

        return clump_dict
