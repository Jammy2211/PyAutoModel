from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Optional, List

if TYPE_CHECKING:
    from autogalaxy.galaxy.galaxy import Galaxy

import autofit as af


logger = logging.getLogger(__name__)


def _galaxies_from(fit: af.Fit, instance: af.ModelInstance) -> List[Galaxy]:
    """
    Returns a list of `Galaxy` objects from a `PyAutoFit` sqlite database `Fit` object.

    The results of a model-fit can be stored in a sqlite database, including the following attributes of the fit:

    - The model and its best fit parameters (e.g. `model.json`).
    - The adapt images associated with adaptive galaxy features (`adapt` folder).

    Each individual attribute can be loaded from the database via the `fit.value()` method.

    This method combines all of these attributes and returns a list of `Galaxy` object for a given non-linear search
    sample (e.g. the maximum likelihood model). This includes associating adapt images with their respective galaxies.

    If multiple `Galaxy` objects were fitted simultaneously via analysis summing, the `fit.child_values()` method
    is instead used to load lists of galaxies. This is necessary if each analysis has different galaxies (e.g. certain
    parameters vary across each dataset and `Analysis` object).

    Parameters
    ----------
    fit
        A `PyAutoFit` `Fit` object which contains the results of a model-fit as an entry in a sqlite database.
    instance
        A manual instance that overwrites the max log likelihood instance in fit (e.g. for drawing the instance
        randomly from the PDF).
    """

    if instance is not None:
        galaxies = instance.galaxies

        if hasattr(instance, "extra_galaxies"):
            if fit.instance.extra_galaxies is not None:
                galaxies = galaxies + fit.instance.extra_galaxies

    else:
        galaxies = fit.instance.galaxies

        if hasattr(fit.instance, "extra_galaxies"):
            if fit.instance.extra_galaxies is not None:
                galaxies = galaxies + fit.instance.extra_galaxies

    if fit.children is not None:
        if len(fit.children) > 0:
            logger.info(
                """
                Using database for a fit with multiple summed Analysis objects.
    
                Galaxy objects do not fully support this yet (e.g. variables across Analysis objects may not be correct)
                so proceed with caution!
                """
            )

            return [galaxies] * len(fit.children)

    return [galaxies]


class GalaxiesAgg(af.AggBase):
    """
    Interfaces with an `PyAutoFit` aggregator object to create instances of `Galaxy` objects from the results
    of a model-fit.

    The results of a model-fit can be stored in a sqlite database, including the following attributes of the fit:

    - The model and its best fit parameters (e.g. `model.json`).
    - The adapt images associated with adaptive galaxy features (`adapt` folder).

    The `aggregator` contains the path to each of these files, and they can be loaded individually. This class
    can load them all at once and create lists of `Galaxy` objects via the `_galaxies_from` method.

    This class's methods returns generators which create the instances of the `Galaxy` objects. This ensures
    that large sets of results can be efficiently loaded from the hard-disk and do not require storing all
    `Galaxy` instances in the memory at once.

    For example, if the `aggregator` contains 3 model-fits, this class can be used to create a generator which
    creates instances of the corresponding 3 `Galaxy` objects.

    If multiple `Galaxy` objects were fitted simultaneously via analysis summing, the `fit.child_values()` method
    is instead used to load lists of galaxies. This is necessary if each analysis has different galaxies (e.g. certain
    parameters vary across each dataset and `Analysis` object).

    This can be done manually, but this object provides a more concise API.

    Parameters
    ----------
    aggregator
        A `PyAutoFit` aggregator object which can load the results of model-fits.
    """

    def object_via_gen_from(
        self, fit, instance: Optional[af.ModelInstance] = None
    ) -> List[Galaxy]:
        """
        Returns a generator of `Galaxy` objects from an input aggregator.

        See `__init__` for a description of how the `Galaxy` objects are created by this method.

        Parameters
        ----------
        fit
            A `PyAutoFit` `Fit` object which contains the results of a model-fit as an entry in a sqlite database.
        instance
            A manual instance that overwrites the max log likelihood instance in fit (e.g. for drawing the instance
            randomly from the PDF).
        """
        return _galaxies_from(fit=fit, instance=instance)
