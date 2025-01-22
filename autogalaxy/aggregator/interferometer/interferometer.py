from functools import partial
from typing import List, Optional

import autofit as af
import autoarray as aa


def _interferometer_from(
    fit: af.Fit,
    real_space_mask: Optional[aa.Mask2D] = None,
) -> List[aa.Interferometer]:
    """
    Returns a list of `Interferometer` objects from a `PyAutoFit` sqlite database `Fit` object.

    The results of a model-fit can be stored in a sqlite database, including the following attributes of the fit:

    - The interferometer visibilities data as a .fits file (`dataset/data.fits`).
    - The visibilities noise-map as a .fits file (`dataset/noise_map.fits`).
    - The uv wavelengths as a .fits file (`dataset/uv_wavelengths.fits`).
    - The real space mask defining the grid of the interferometer for the FFT (`dataset/real_space_mask.fits`).
    - The settings of the `Interferometer` data structure used in the fit (`dataset/settings.json`).

    Each individual attribute can be loaded from the database via the `fit.value()` method.

    This method combines all of these attributes and returns a `Interferometer` object, including having its
    settings updated to the values used by the model-fit.

    If multiple `Interferometer` objects were fitted simultaneously via analysis summing, the `fit.child_values()`
    method is instead used to load lists of the data, noise-map, PSF and mask and combine them into a list of
    `Interferometer` objects.

    Parameters
    ----------
    fit
        A `PyAutoFit` `Fit` object which contains the results of a model-fit as an entry in a sqlite database.
    """

    fit_list = [fit] if not fit.children else fit.children

    dataset_list = []

    for fit in fit_list:
        data = aa.Visibilities(
            visibilities=fit.value(name="dataset.data").data.astype("float")
        )
        noise_map = aa.VisibilitiesNoiseMap(
            fit.value(name="dataset.noise_map").data.astype("float")
        )
        uv_wavelengths = fit.value(name="dataset.uv_wavelengths").data

        real_space_mask = (
            real_space_mask
            if real_space_mask is not None
            else aa.Mask2D.from_primary_hdu(
                primary_hdu=fit.value(name="dataset.real_space_mask")
            )
        )

        transformer_class = fit.value(name="dataset.transformer_class")

        dataset = aa.Interferometer(
            data=data,
            noise_map=noise_map,
            uv_wavelengths=uv_wavelengths,
            real_space_mask=real_space_mask,
            transformer_class=transformer_class,
        )

        dataset_list.append(dataset)

    return dataset_list


class InterferometerAgg:
    def __init__(self, aggregator: af.Aggregator):
        """
        Interfaces with an `PyAutoFit` aggregator object to create instances of `Interferometer` objects from the results
        of a model-fit.

        The results of a model-fit can be stored in a sqlite database, including the following attributes of the fit:

        - The interferometer visibilities data as a .fits file (`dataset/data.fits`).
        - The visibilities noise-map as a .fits file (`dataset/noise_map.fits`).
        - The uv wavelengths as a .fits file (`dataset/uv_wavelengths.fits`).
        - The real space mask defining the grid of the interferometer for the FFT (`dataset/real_space_mask.fits`).
        - The settings of the `Interferometer` data structure used in the fit (`dataset/settings.json`).

        The `aggregator` contains the path to each of these files, and they can be loaded individually. This class
        can load them all at once and create an `Interferometer` object via the `_interferometer_from` method.

        This class's methods returns generators which create the instances of the `Interferometer` objects. This ensures
        that large sets of results can be efficiently loaded from the hard-disk and do not require storing all
        `Interferometer` instances in the memory at once.

        For example, if the `aggregator` contains 3 model-fits, this class can be used to create a generator which
        creates instances of the corresponding 3 `Interferometer` objects.

        If multiple `Interferometer` objects were fitted simultaneously via analysis summing, the `fit.child_values()`
        method is instead used to load lists of the data, noise-map, PSF and mask and combine them into a list of
        `Interferometer` objects.

        This can be done manually, but this object provides a more concise API.

        Parameters
        ----------
        aggregator
            A `PyAutoFit` aggregator object which can load the results of model-fits.
        """
        self.aggregator = aggregator

    def dataset_gen_from(
        self,
        real_space_mask: Optional[aa.Mask2D] = None,
    ) -> List[aa.Interferometer]:
        """
        Returns a generator of `Interferometer` objects from an input aggregator.

        See `__init__` for a description of how the `Interferometer` objects are created by this method.

        Parameters
        ----------
        real_space_mask
            The real space mask.
        """
        func = partial(
            _interferometer_from,
            real_space_mask=real_space_mask,
        )

        return self.aggregator.map(func=func)
