from functools import partial
from typing import List

import autofit as af
import autoarray as aa


def _imaging_from(
    fit: af.Fit,
) -> List[aa.Imaging]:
    """
    Returns a list of `Imaging` objects from a `PyAutoFit` sqlite database `Fit` object.

    The results of a model-fit can be stored in a sqlite database, including the following attributes of the fit:

    - The imaging data as a .fits file (`dataset/data.fits`).
    - The noise-map as a .fits file (`dataset/noise_map.fits`).
    - The point spread function as a .fits file (`dataset/psf.fits`).
    - The settings of the `Imaging` data structure used in the fit (`dataset/settings.json`).
    - The mask used to mask the `Imaging` data structure in the fit (`dataset/mask.fits`).

    Each individual attribute can be loaded from the database via the `fit.value()` method.

    This method combines all of these attributes and returns a `Imaging` object, has the mask applied to the
    `Imaging` data structure and its settings updated to the values used by the model-fit.

    If multiple `Imaging` objects were fitted simultaneously via analysis summing, the `fit.child_values()` method
    is instead used to load lists of the data, noise-map, PSF and mask and combine them into a list of
    `Imaging` objects.

    Parameters
    ----------
    fit
        A `PyAutoFit` `Fit` object which contains the results of a model-fit as an entry in a sqlite database.
    """

    fit_list = [fit] if not fit.children else fit.children

    dataset_list = []

    for fit in fit_list:
        data = aa.Array2D.from_primary_hdu(primary_hdu=fit.value(name="dataset.data"))
        noise_map = aa.Array2D.from_primary_hdu(
            primary_hdu=fit.value(name="dataset.noise_map")
        )
        try:
            psf = aa.Kernel2D.from_primary_hdu(
                primary_hdu=fit.value(name="dataset.psf")
            )
        except AttributeError:
            psf = None

        dataset = aa.Imaging(
            data=data,
            noise_map=noise_map,
            psf=psf,
            check_noise_map=False,
        )

        mask = aa.Mask2D.from_primary_hdu(primary_hdu=fit.value(name="dataset.mask"))

        dataset = dataset.apply_mask(mask=mask)

        try:
            over_sample_size_lp = aa.Array2D.from_primary_hdu(
                primary_hdu=fit.value(name="dataset.over_sample_size_lp")
            ).native
            over_sample_size_lp = over_sample_size_lp.apply_mask(mask=mask)
        except AttributeError:
            over_sample_size_lp = 1

        try:
            over_sample_size_pixelization = aa.Array2D.from_primary_hdu(
                primary_hdu=fit.value(name="dataset.over_sample_size_pixelization")
            ).native
            over_sample_size_pixelization = over_sample_size_pixelization.apply_mask(
                mask=mask
            )
        except AttributeError:
            over_sample_size_pixelization = 1

        dataset = dataset.apply_over_sampling(
            over_sample_size_lp=over_sample_size_lp,
            over_sample_size_pixelization=over_sample_size_pixelization,
        )

        dataset_list.append(dataset)

    return dataset_list


class ImagingAgg:
    def __init__(self, aggregator: af.Aggregator):
        """
        Interfaces with an `PyAutoFit` aggregator object to create instances of `Imaging` objects from the results
        of a model-fit.

        The results of a model-fit can be stored in a sqlite database, including the following attributes of the fit:

        - The imaging data as a .fits file (`dataset/data.fits`).
        - The noise-map as a .fits file (`dataset/noise_map.fits`).
        - The point spread function as a .fits file (`dataset/psf.fits`).
        - The settings of the `Imaging` data structure used in the fit (`dataset/settings.json`).
        - The mask used to mask the `Imaging` data structure in the fit (`dataset/mask.fits`).

        The `aggregator` contains the path to each of these files, and they can be loaded individually. This class
        can load them all at once and create an `Imaging` object via the `_imaging_from` method.

        This class's methods returns generators which create the instances of the `Imaging` objects. This ensures
        that large sets of results can be efficiently loaded from the hard-disk and do not require storing all
        `Imaging` instances in the memory at once.

        For example, if the `aggregator` contains 3 model-fits, this class can be used to create a generator which
        creates instances of the corresponding 3 `Imaging` objects.

        If multiple `Imaging` objects were fitted simultaneously via analysis summing, the `fit.child_values()` method
        is instead used to load lists of the data, noise-map, PSF and mask and combine them into a list of
        `Imaging` objects.

        This can be done manually, but this object provides a more concise API.

        Parameters
        ----------
        aggregator
            A `PyAutoFit` aggregator object which can load the results of model-fits.
        """
        self.aggregator = aggregator

    def dataset_gen_from(
        self,
    ) -> List[aa.Imaging]:
        """
        Returns a generator of `Imaging` objects from an input aggregator.

        See `__init__` for a description of how the `Imaging` objects are created by this method.
        """

        func = partial(_imaging_from)

        return self.aggregator.map(func=func)
