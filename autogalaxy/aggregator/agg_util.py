from __future__ import annotations
from typing import List, Optional

import autofit as af
import autoarray as aa

from autogalaxy.analysis.adapt_images.adapt_images import AdaptImages


def adapt_images_from(
    fit: af.Fit,
) -> List[AdaptImages]:
    """
    Updates adaptive images when loading the galaxies from a `PyAutoFit` sqlite database `Fit` object.

    This function ensures that if adaptive features (e.g. an `Hilbert` image-mesh) are used in a model-fit,
    they work when using the database to load and inspect the results of the model-fit.

    Parameters
    ----------
    fit
        A `PyAutoFit` `Fit` object which contains the results of a model-fit as an entry in a sqlite database.
    galaxies
        A list of galaxies corresponding to a sample of a non-linear search and model-fit.

    Returns
    -------
    A list of galaxies associated with a sample of the non-linear search with adaptive images associated with them.
    """

    fit_list = [fit] if not fit.children else fit.children

    if fit.value(name="adapt_images.adapt_images") is None:
        return [None] * len(fit_list)

    adapt_images_list = []

    for fit in fit_list:
        try:
            mask = aa.Mask2D.from_primary_hdu(
                primary_hdu=fit.value(name="dataset.mask")
            )
        except AttributeError:
            mask = aa.Mask2D.from_primary_hdu(
                primary_hdu=fit.value(name="dataset.real_space_mask")
            )

        galaxy_name_image_dict = {}

        adapt_image_name_list = fit.value(name="adapt_images.adapt_images")

        for name in adapt_image_name_list:
            adapt_image = aa.Array2D.from_primary_hdu(
                primary_hdu=fit.value(name=f"adapt_images.{name}")
            )
            adapt_image = adapt_image.apply_mask(mask=mask)
            galaxy_name_image_dict[name] = adapt_image

        instance = fit.model.instance_from_prior_medians(ignore_prior_limits=True)

        adapt_images = AdaptImages(galaxy_name_image_dict=galaxy_name_image_dict)

        adapt_images = adapt_images.updated_via_instance_from(
            instance=instance,
            mask=mask,
        )

        adapt_images_list.append(adapt_images)

    return adapt_images_list
