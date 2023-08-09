from __future__ import annotations
from typing import TYPE_CHECKING, Optional, List

if TYPE_CHECKING:
    from autogalaxy.galaxy.galaxy import Galaxy
    from autogalaxy.interferometer.fit_interferometer import FitInterferometer

import autofit as af
import autoarray as aa

from autogalaxy.aggregator.abstract import AbstractAgg
from autogalaxy.analysis.preloads import Preloads

from autogalaxy.aggregator import agg_util
from autogalaxy.aggregator.interferometer import _interferometer_from
from autogalaxy.aggregator.plane import _plane_from


def _fit_interferometer_from(
    fit: af.Fit,
    galaxies: List[Galaxy],
    real_space_mask: Optional[aa.Mask2D] = None,
    settings_dataset: aa.SettingsInterferometer = None,
    settings_pixelization: aa.SettingsPixelization = None,
    settings_inversion: aa.SettingsInversion = None,
    use_preloaded_grid: bool = True,
) -> List[FitInterferometer]:
    """
    Returns a list of `FitInterferometer` objects from a `PyAutoFit` sqlite database `Fit` object.

    The results of a model-fit can be stored in a sqlite database, including the following attributes of the fit:

    - The interferometer data, noise-map, uv-wavelengths and settings as .fits files (e.g. `dataset/data.fits`).
    - The real space mask defining the grid of the interferometer for the FFT (`dataset/real_space_mask.fits`).
    - The settings of pixelization used by the fit (`dataset/settings_pixelization.json`).
    - The settings of inversions used by the fit (`dataset/settings_inversion.json`).

    Each individual attribute can be loaded from the database via the `fit.value()` method.

    This method combines all of these attributes and returns a `FitInterferometer` object for a given non-linear
    search sample (e.g. the maximum likelihood model). This includes associating adapt images with their respective
    galaxies.

    If multiple `FitInterferometer` objects were fitted simultaneously via analysis summing, the `fit.child_values()`
    method is instead used to load lists of the data, noise-map, PSF and mask and combine them into a list of
    `FitInterferometer` objects.

    The settings of a pixelization of inversion can be overwritten by inputting a `settings_dataset` object, for
    example if you want to use a grid with a different inversion solver.

    Parameters
    ----------
    fit
        A `PyAutoFit` `Fit` object which contains the results of a model-fit as an entry in a sqlite database.
    galaxies
        A list of galaxies corresponding to a sample of a non-linear search and model-fit.
    settings_dataset
        Optionally overwrite the `SettingsInterferometer` of the `Interferometer` object that is created from the fit.
    settings_pixelization
        Optionally overwrite the `SettingsPixelization` of the `Pixelization` object that is created from the fit.
    settings_inversion
        Optionally overwrite the `SettingsInversion` of the `Inversion` object that is created from the fit.
    use_preloaded_grid
        Certain pixelization's construct their mesh in the source-plane from a stochastic KMeans algorithm. This grid
        may be output to hard-disk after the model-fit and loaded via the database to ensure the same grid is used
        as the fit.
    """
    from autogalaxy.interferometer.fit_interferometer import FitInterferometer

    dataset_list = _interferometer_from(
        fit=fit,
        real_space_mask=real_space_mask,
        settings_dataset=settings_dataset,
    )
    plane = _plane_from(fit=fit, galaxies=galaxies)

    settings_pixelization = settings_pixelization or fit.value(
        name="settings_pixelization"
    )
    settings_inversion = settings_inversion or fit.value(name="settings_inversion")

    sparse_grids_of_planes_list = agg_util.sparse_grids_of_planes_list_from(
        fit=fit, total_fits=len(dataset_list), use_preloaded_grid=use_preloaded_grid
    )

    preloads = None

    fit_dataset_list = []

    for dataset, sparse_grids_of_planes in zip(dataset_list, sparse_grids_of_planes_list):

        if use_preloaded_grid:

            if sparse_grids_of_planes is not None:
                preloads = Preloads(
                    sparse_image_plane_grid_pg_list=sparse_grids_of_planes,
                )

                if len(preloads.sparse_image_plane_grid_pg_list) == 2:
                    if type(preloads.sparse_image_plane_grid_pg_list[1]) != list:
                        preloads.sparse_image_plane_grid_pg_list[1] = [
                            preloads.sparse_image_plane_grid_pg_list[1]
                        ]

        fit_dataset_list.append(FitInterferometer(
            dataset=dataset,
            plane=plane,
            settings_pixelization=settings_pixelization,
            settings_inversion=settings_inversion,
            preloads=preloads,
        )
    )

    return fit_dataset_list


class FitInterferometerAgg(AbstractAgg):
    def __init__(
        self,
        aggregator: af.Aggregator,
        settings_dataset: Optional[aa.SettingsInterferometer] = None,
        settings_pixelization: Optional[aa.SettingsPixelization] = None,
        settings_inversion: Optional[aa.SettingsInversion] = None,
        use_preloaded_grid: bool = True,
        real_space_mask: Optional[aa.Mask2D] = None,
    ):
        """
        Interfaces with an `PyAutoFit` aggregator object to create instances of `FitInterferometer` objects from the
        results of a model-fit.

        The results of a model-fit can be stored in a sqlite database, including the following attributes of the fit:

        - The interferometer data, noise-map, uv-wavelengths and settings as .fits files (e.g. `dataset/data.fits`).
        - The real space mask defining the grid of the interferometer for the FFT (`dataset/real_space_mask.fits`).
        - The settings of pixelization used by the fit (`dataset/settings_pixelization.json`).
        - The settings of inversions used by the fit (`dataset/settings_inversion.json`).

        The `aggregator` contains the path to each of these files, and they can be loaded individually. This class
        can load them all at once and create an `FitInterferometer` object via the `_fit_interferometer_from` method.

        This class's methods returns generators which create the instances of the `FitInterferometer` objects. This ensures
        that large sets of results can be efficiently loaded from the hard-disk and do not require storing all
        `FitInterferometer` instances in the memory at once.

        For example, if the `aggregator` contains 3 model-fits, this class can be used to create a generator which
        creates instances of the corresponding 3 `FitInterferometer` objects.

        If multiple `Imaging` objects were fitted simultaneously via analysis summing, the `fit.child_values()` method
        is instead used to load lists of the data, noise-map, PSF and mask and combine them into a list of
        `FitImaging` objects.

        This can be done manually, but this object provides a more concise API.

        Parameters
        ----------
        aggregator
            A `PyAutoFit` aggregator object which can load the results of model-fits.
        settings_dataset
            Optionally overwrite the `SettingsInterferometer` of the `Interferometer` object that is created from the fit.
        settings_pixelization
            Optionally overwrite the `SettingsPixelization` of the `Pixelization` object that is created from the fit.
        settings_inversion
            Optionally overwrite the `SettingsInversion` of the `Inversion` object that is created from the fit.
        use_preloaded_grid
            Certain pixelization's construct their mesh in the source-plane from a stochastic KMeans algorithm. This
            grid may be output to hard-disk after the model-fit and loaded via the database to ensure the same grid is
            used as the fit.
        """
        super().__init__(aggregator=aggregator)

        self.settings_dataset = settings_dataset
        self.settings_pixelization = settings_pixelization
        self.settings_inversion = settings_inversion
        self.use_preloaded_grid = use_preloaded_grid
        self.real_space_mask = real_space_mask

    def object_via_gen_from(self, fit, galaxies) -> List[FitInterferometer]:
        """
        Returns a generator of `FitInterferometer` objects from an input aggregator.

        See `__init__` for a description of how the `FitInterferometer` objects are created by this method.

        Parameters
        ----------
        fit
            A `PyAutoFit` `Fit` object which contains the results of a model-fit as an entry in a sqlite database.
        galaxies
            A list of galaxies corresponding to a sample of a non-linear search and model-fit.
        """
        return _fit_interferometer_from(
            fit=fit,
            galaxies=galaxies,
            settings_dataset=self.settings_dataset,
            settings_pixelization=self.settings_pixelization,
            settings_inversion=self.settings_inversion,
            use_preloaded_grid=self.use_preloaded_grid,
        )
