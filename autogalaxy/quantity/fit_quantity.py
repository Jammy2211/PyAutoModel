from typing import List, Optional, Union

import autoarray as aa

from autogalaxy.quantity.dataset_quantity import DatasetQuantity
from autogalaxy.profiles.light.abstract import LightProfile
from autogalaxy.profiles.mass.abstract.abstract import MassProfile
from autogalaxy.galaxy.galaxy import Galaxy
from autogalaxy.galaxy.galaxies import Galaxies


class FitQuantity(aa.FitImaging):
    def __init__(
        self,
        dataset: DatasetQuantity,
        light_mass_obj: Union[LightProfile, MassProfile, Galaxy, Galaxies],
        func_str: str,
        model_data_manual: Optional[Union[aa.Array2D, aa.VectorYX2D]] = None,
    ):
        """
        Fits a `DatasetQuantity` object with model data.

        This is used to fit a quantity (e.g. a convergence, deflection angles), from an object like
        a `LightProfile`, `MassProfile`, `Galaxy`, to the same quantity derived from another of that object.

        For example, we may have the 2D convergence of a power-law mass profile and wish to determine how closely the
        2D convergence of an nfw mass profile's matches it. The `FitQuantity` can fit the two, where a noise-map
        is associated with the quantity's dataset such that figure of merits like a chi-squared and log likelihood
        can be computed.

        This is ultimately used in the `AnalysisQuantity` class to perform model-fitting of quantities of different
        mass profiles, light profiles, galaxies, etc.

        Parameters
        ----------
        dataset
            The quantity that is to be fitted, which has a noise-map associated it with for computing goodness-of-fit
            metrics.
        light_mass_obj
            An object containing functions which computes a light and / or mass quantity (e.g. galaxies)
            whose model quantities are used to fit the quantity data.
        func_str
            A string giving the name of the method of the input galaxy used to compute the quantity that fits
            the dataset.
        model_data_manual
            Manually pass the model-data, omitting its calculation via the function defined by the `func_str`.
        """

        self.light_mass_obj = light_mass_obj
        self.func_str = func_str
        self.model_data_manual = model_data_manual

        super().__init__(dataset=dataset, use_mask_in_fit=False)

    @property
    def model_data(self):
        if self.model_data_manual is None:
            func = getattr(self.light_mass_obj, self.func_str)
            return func(grid=self.grids.lp)

        return self.model_data_manual

    @property
    def y(self) -> "FitQuantity":
        """
        If the `FitQuantity` contains a `VectorYX2D` as its data, this property returns a new `FitQuantity`
        with just the y-values of the vectors as the data, alongside the noise-map. The y values of the model-data are
        also extracted and used in this fit.

        This is primarily used for visualizing a fit to the `FitQuantity` containing vectors, as it allows one to
        reuse tools which visualize `Array2D` objects.
        """
        if isinstance(self.data, aa.VectorYX2D):
            return FitQuantity(
                dataset=self.dataset.y,
                light_mass_obj=self.light_mass_obj,
                func_str=self.func_str,
                model_data_manual=self.model_data.y,
            )

    @property
    def x(self) -> "FitQuantity":
        """
        If the `FitQuantity` contains a `VectorYX2D` as its data, this property returns a new `FitQuantity`
        with just the x-values of the vectors as the data, alongside the noise-map. The x values of the model-data are
        also extracted and used in this fit.

        This is primarily used for visualizing a fit to the `FitQuantity` containing vectors, as it allows one to
        reuse tools which visualize `Array2D` objects.
        """
        if isinstance(self.data, aa.VectorYX2D):
            return FitQuantity(
                dataset=self.dataset.x,
                light_mass_obj=self.light_mass_obj,
                func_str=self.func_str,
                model_data_manual=self.model_data.x,
            )

    @property
    def mask(self):
        return self.dataset.mask

    @property
    def inversion(self):
        return None

    @property
    def galaxies(self):
        return self.light_mass_obj
