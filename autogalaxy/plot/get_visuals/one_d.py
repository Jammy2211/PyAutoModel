import logging
from typing import List, Union

import autoarray as aa
import autoarray.plot as aplt

from autogalaxy.galaxy.galaxy import Galaxy
from autogalaxy.plot.include.one_d import Include1D
from autogalaxy.plot.visuals.one_d import Visuals1D
from autogalaxy.profiles.light.abstract import LightProfile
from autogalaxy.profiles.mass.abstract.abstract import MassProfile

from autogalaxy import exc
from autogalaxy.util import error_util

logger = logging.getLogger(__name__)


class GetVisuals1D(aplt.GetVisuals1D):
    def __init__(self, include: Include1D, visuals: Visuals1D):
        """
        Class which gets 1D attributes and adds them to a `Visuals1D` objects, such that they are plotted on 1D figures.

        For a visual to be extracted and added for plotting, it must have a `True` value in its corresponding entry in
        the `Include1D` object. If this entry is `False`, the `GetVisuals1D.get` method returns a None and the attribute
        is omitted from the plot.

        The `GetVisuals1D` class adds new visuals to a pre-existing `Visuals1D` object that is passed to its `__init__`
        method. This only adds a new entry if the visual are not already in this object.

        Parameters
        ----------
        include
            Sets which 1D visuals are included on the figure that is to be plotted (only entries which are `True`
            are extracted via the `GetVisuals1D` object).
        visuals
            The pre-existing visuals of the plotter which new visuals are added too via the `GetVisuals1D` class.
        """
        super().__init__(include=include, visuals=visuals)

    def via_light_obj_from(self, light_obj: Union[LightProfile, Galaxy]) -> Visuals1D:
        """
        From an object with light profiles (e.g. a `LightProfile`, `Galaxy`) get its attributes that can be plotted
        and return them  in a `Visuals1D` object.

        Only attributes not already in `self.visuals` and with `True` entries in the `Include1D` object are extracted
        for plotting.

        From a light object the following 1D attributes can be extracted for plotting:

        - half_light_radius: the radius containing 50% of the light objects total integrated luminosity.

        Parameters
        ----------
        light_obj
            The light object (e.g. a `LightProfile`, `Galaxy`) whose attributes are extracted for plotting.

        Returns
        -------
        Visuals1D
            The collection of attributes that can be plotted by a `Plotter` object.
        """

        half_light_radius = self.get(
            "half_light_radius", value=light_obj.half_light_radius
        )

        return self.visuals + self.visuals.__class__(
            half_light_radius=half_light_radius
        )

    def via_light_obj_list_from(
        self, light_obj_list: Union[List[LightProfile], List[Galaxy]], low_limit: float
    ) -> Visuals1D:
        """
        From a list of objects with light profiles (e.g. a `LightProfile`, `Galaxy`) get its attributes that can be
        plotted and return them  in a `Visuals1D` object.

        Only attributes not already in `self.visuals` and with `True` entries in the `Include1D` object are extracted
        for plotting.

        This function iterates over all light objects in the list and averages over each attribute's values to estimate
        the mean value of the attribute and its error, both of which can then be plotted. This is typically used
        to plot 1D errors on a quantity that are estimated via a Probability Density Function.

        From a light object lust the following 1D attributes can be extracted for plotting:

        - half_light_radius: the radius containing 50% of the light objects total integrated luminosity.

        Parameters
        ----------
        light_obj_list
            The list of light objects (e.g. a `LightProfile`, `Galaxy`) whose mean attributes and error estimates are
            extracted for plotting.
        low_limit
            The value of sigma to which errors are estimated (e.g. 1.0 will estimate errors at the ~0.32 and ~0.68
            intervals of the probability distribution.

        Returns
        -------
        Visuals1D
            The mean value and errors of each attribute that are plotted in 1D by a `Plotter` object.
        """

        if self.include.half_light_radius:

            half_light_radius_list = [
                light_profile.half_light_radius for light_profile in light_obj_list
            ]

            if None in half_light_radius_list:

                half_light_radius = None
                half_light_radius_errors = None

            else:

                (
                    half_light_radius,
                    half_light_radius_errors,
                ) = error_util.value_median_and_error_region_via_quantile(
                    value_list=half_light_radius_list, low_limit=low_limit
                )

        else:

            half_light_radius = None
            half_light_radius_errors = None

        half_light_radius = self.get("half_light_radius", value=half_light_radius)
        half_light_radius_errors = self.get(
            "half_light_radius", value=half_light_radius_errors
        )

        return self.visuals + self.visuals.__class__(
            half_light_radius=half_light_radius,
            half_light_radius_errors=half_light_radius_errors,
        )

    def via_mass_obj_from(
        self, mass_obj: Union[MassProfile, Galaxy], grid: aa.type.Grid2DLike
    ) -> Visuals1D:
        """
        From an object with mass profiles (e.g. a `MassProfile`, `Galaxy`) get its attributes that can be plotted
        and return them  in a `Visuals1D` object.

        Only attributes not already in `self.visuals` and with `True` entries in the `Include1D` object are extracted
        for plotting.

        From a mass object the following 1D attributes can be extracted for plotting:

        - einstein_radius: the einstein radius (e.g. area within critical curve) of the mass object.

        Mass profiles can be too shallow to do lensing and therefore an Einstein radius cannot be computed. This
        raises a TypeError which is accounted for below.

        Parameters
        ----------
        mass_obj
            The mass object (e.g. a `MassProfile`, `Galaxy`) whose attributes are extracted for plotting.

        Returns
        -------
        Visuals1D
            The collection of attributes that can be plotted by a `Plotter` object.
        """

        einstein_radius = None

        if self.include.einstein_radius:
            try:
                einstein_radius = mass_obj.einstein_radius_from(grid=grid)
            except (TypeError, AttributeError):
                pass

        einstein_radius = self.get("einstein_radius", value=einstein_radius)

        return self.visuals + self.visuals.__class__(einstein_radius=einstein_radius)

    def via_mass_obj_list_from(
        self,
        mass_obj_list: Union[List[MassProfile], List[Galaxy]],
        grid: aa.type.Grid2DLike,
        low_limit: float,
    ) -> Visuals1D:
        """
        From a list of objects with mass profiles (e.g. a `MassProfile`, `Galaxy`) get its attributes that can be
        plotted and return them  in a `Visuals1D` object.

        Only attributes not already in `self.visuals` and with `True` entries in the `Include1D` object are extracted
        for plotting.

        This function iterates over all mass objects in the list and averages over each attribute's values to estimate
        the mean value of the attribute and its error, both of which can then be plotted. This is typically used
        to plot 1D errors on a quantity that are estimated via a Probability Density Function.

        From a mass object lust the following 1D attributes can be extracted for plotting:

        - half_mass_radius: the radius containing 50% of the mass objects total integrated luminosity.

        Parameters
        ----------
        mass_obj_list
            The list of mass objects (e.g. a `MassProfile`, `Galaxy`) whose mean attributes and error estimates are
            extracted for plotting.
        low_limit
            The value of sigma to which errors are estimated (e.g. 1.0 will estimate errors at the ~0.32 and ~0.68
            intervals of the probability distribution.

        Returns
        -------
        Visuals1D
            The mean value and errors of each attribute that are plotted in 1D by a `Plotter` object.
        """

        if self.include.einstein_radius:

            einstein_radius_list = []

            for mass_obj in mass_obj_list:

                try:
                    einstein_radius_list.append(
                        mass_obj.einstein_radius_from(grid=grid)
                    )
                except TypeError:
                    einstein_radius_list.append(None)

            einstein_radius_list = list(filter(None, einstein_radius_list))

            (
                einstein_radius,
                einstein_radius_errors,
            ) = error_util.value_median_and_error_region_via_quantile(
                value_list=einstein_radius_list, low_limit=low_limit
            )

        else:

            einstein_radius = None
            einstein_radius_errors = None

        einstein_radius = self.get("einstein_radius", value=einstein_radius)
        einstein_radius_errors = self.get(
            "einstein_radius", value=einstein_radius_errors
        )

        return self.visuals + self.visuals.__class__(
            einstein_radius=einstein_radius,
            einstein_radius_errors=einstein_radius_errors,
        )
