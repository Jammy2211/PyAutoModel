from __future__ import annotations
import logging
from typing import List

import autofit as af
import autoarray as aa

logger = logging.getLogger(__name__)


def _dataset_model_from(
    fit: af.Fit, instance: af.ModelInstance
) -> List[aa.DatasetModel]:
    """
    Returns a `DatasetModel` object from a `PyAutoFit` sqlite database `Fit` object.

    The results of a model-fit can be stored in a sqlite database, including the following attributes of the fit:

    - The model and its best fit parameters (e.g. `model.json`).

    Each individual attribute can be loaded from the database via the `fit.value()` method.

    This method combines all of these attributes and returns a `DatasetModel` object for a given non-linear search
    sample (e.g. the maximum likelihood model).

    If multiple `DatasetModel` objects were fitted simultaneously via analysis summing, the `fit.child_values()` method
    is instead used to load lists of dataset models. This is necessary if each analysis has different dataset
    models (e.g. certain parameters vary across each dataset and `Analysis` object).

    Parameters
    ----------
    fit
        A `PyAutoFit` `Fit` object which contains the results of a model-fit as an entry in a sqlite database.
    instance
        A manual instance that overwrites the max log likelihood instance in fit (e.g. for drawing the instance
        randomly from the PDF).
    """

    if instance is not None:
        try:
            dataset_model = instance.dataset_model
        except AttributeError:
            dataset_model = None
    else:
        try:
            dataset_model = fit.instance.dataset_model
        except AttributeError:
            dataset_model = None

    if fit.children is not None:
        if len(fit.children) > 0:
            logger.info(
                """
                Using database for a fit with multiple summed Analysis objects.
    
                DatasetModel objects do not fully support this yet (e.g. variables across Analysis objects may not be correct)
                so proceed with caution!
                """
            )

            return [dataset_model] * len(fit.children)

    return [dataset_model]
