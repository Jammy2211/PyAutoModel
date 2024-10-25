from typing import List, Optional, Type

import autofit as af
import autoarray as aa

from autogalaxy.galaxy.galaxy import Galaxy
from autogalaxy.profiles.light.abstract import LightProfile
from autogalaxy.profiles.mass.abstract.abstract import MassProfile


class ClumpModel:
    def __init__(
        self,
        redshift: float,
        centres: aa.Grid2DIrregular,
        light_cls: Optional[Type[LightProfile]] = None,
        mass_cls: Optional[Type[MassProfile]] = None,
        einstein_radius_upper_limit: Optional[float] = None,
        unfix_centres: bool = False,
    ):
        """
        The clump API allows creates model components which model the light and mass of galaxies that are nearby the
        main galaxy(s) of interest.

        The `ClumpModel` object handles the creation of these model components to streamline model composition with
        multiple clumps.

        Every galaxy which is modeled as a clump has its centre input into this object which is fixed to this value
        for model-fitting. All clumps are created as model `Galaxy` objects with a shard input redshift.

        The light and mass profiles of the clumps are input via the `light_cls` and `mass_cls` inputs. If either is
        omitted the clumps are not assigned a light or mass model.

        Parameters
        ----------
        redshift
            The redshift value of all clumps, which is likely the same as the main galaxy redshift.
        centres
            The centre of every clump in the model, whose light and mass profile centres are fixed to this value
            throughout the model-fit.
        light_cls
            The light profile given to all clumps; if omitted all clumps have no light profile.
        mass_cls
            The mass profile given to all clumps; if omitted all clumps have no mass profile.
        einstein_radius_upper_limit
            The upper limit given to any mass model's `einstein_radius` parameter (e.g. if `IsothermalSph` profiles
            are used to model clumps).
        unfix_centres
            If required, change the mass and light centres from fixed values to Uniform Prior Models +/- 0.1 around
            the input centres.
        """
        self.redshift = redshift
        self.centres = centres

        self.light_cls = light_cls
        self.mass_cls = mass_cls

        self.einstein_radius_upper_limit = einstein_radius_upper_limit
        self.unfix_centres = unfix_centres

        self.centre_prior_half_width = 0.1

    @property
    def total_clumps(self) -> int:
        return len(self.centres.in_list)

    def unfix_centre(obj, centre):
        obj.centre.centre_0 = af.UniformPrior(
            lower_limit=new_centre[0] - centre_prior_half_width,
            upper_limit=new_centre[0] + centre_prior_half_width,
        )
        obj.centre.centre_1 = af.UniformPrior(
            lower_limit=new_centre[1] - centre_prior_half_width,
            upper_limit=new_centre[1] + centre_prior_half_width,
        )
        return obj

    @property
    def light_list(self) -> Optional[List[af.Model]]:
        """
        Returns a list of every clump's light model, where the centre of that light model is fixed to its corresponding
        input clump's centre, unless specified to be free.
        """
        if self.light_cls is None:
            return None

        light_list = []

        for centre in self.centres.in_list:
            if self.unfix_centres:
                light_list.append(unfix_centre(af.Model(self.light_cls), centre))
            else:
                light_list.append(af.Model(self.light_cls, centre=centre))

        return light_list

    @property
    def mass_list(self) -> Optional[List[af.Model]]:
        """
        Returns a list of every clump's mass model, where the centre of that mass model is fixed to its corresponding
        input clump's centre, unless specified to be free.
        """
        if self.mass_cls is None:
            return None

        mass_list = []

        for centre in self.centres.in_list:
            if self.unfix_centres:
                mass = unfix_centres(af.Model(self.mass_cls), centre)
            else:
                mass = af.Model(self.mass_cls, centre=centre)

            if (
                hasattr(mass, "einstein_radius")
                and self.einstein_radius_upper_limit is not None
            ):
                mass.einstein_radius = af.UniformPrior(
                    lower_limit=0.0, upper_limit=self.einstein_radius_upper_limit
                )
            mass_list.append(mass)

        return mass_list

    @property
    def clumps_light_only(self) -> af.Collection:
        """
        Returns all clumps as a `Collection` model object, which can be easily added to an overall model `Collection`
        of galaxies (see the `clumps` property below for more details).

        All clumps only contain their model mass profiles, which is important for certain pipelines which omit
        the clump mass profiles.
        """
        clumps_light_only = {}

        for i in range(self.total_clumps):
            light = self.light_list[i] if self.light_cls is not None else None

            clumps_light_only[f"clump_{i}"] = af.Model(
                Galaxy, redshift=self.redshift, light=light
            )

        return af.Collection(**clumps_light_only)

    @property
    def clumps_mass_only(self) -> af.Collection:
        """
        Returns all clumps as a `Collection` model object, which can be easily added to an overall model `Collection`
        of galaxies (see the `clumps` property below for more details).

        All clumps only contain their model mass profiles, which is important for certain pipelines which omit
        the clump light profiles.
        """
        clumps_mass_only = {}

        for i in range(self.total_clumps):
            mass = self.mass_list[i] if self.mass_cls is not None else None

            clumps_mass_only[f"clump_{i}"] = af.Model(
                Galaxy, redshift=self.redshift, mass=mass
            )

        return af.Collection(**clumps_mass_only)

    @property
    def clumps(self) -> af.Collection:
        """
        Returns all clumps as a `Collection` model object, which can be easily added to an overall model `Collection`
        of galaxies.

        To make this `Collection` every clump centre and the input `light_cls` / `mass_cls` are used to create a
        dictionary of model `Galaxy` objects with fixed light and mass profile centres. Their redshifts use the
        input redshift.

        The keys of this dictionary are numerically ordered as `clump_0`, `clump_2` etc.
        """
        clumps = {}

        for i in range(self.total_clumps):
            light = self.light_list[i] if self.light_cls is not None else None
            mass = self.mass_list[i] if self.mass_cls is not None else None

            clumps[f"clump_{i}"] = af.Model(
                Galaxy, redshift=self.redshift, light=light, mass=mass
            )

        return af.Collection(**clumps)


class ClumpModelDisabled:
    def __init__(self):
        pass

    @property
    def clumps_light_only(self):
        return af.Collection()

    @property
    def clumps_mass_only(self):
        return af.Collection()

    @property
    def clumps(self):
        return af.Collection()
