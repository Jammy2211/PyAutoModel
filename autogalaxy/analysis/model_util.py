from __future__ import annotations
import logging
import numpy as np
from scipy.stats import norm
from typing import TYPE_CHECKING, ClassVar, Dict, List

if TYPE_CHECKING:
    from autogalaxy.analysis.analysis import AnalysisDataset
    from autogalaxy.analysis.result import ResultDataset

import autofit as af
import autoarray as aa

from autogalaxy.profiles.light.abstract import LightProfile
from autogalaxy.profiles.mass.abstract.abstract import MassProfile

from autogalaxy import exc

logger = logging.getLogger(__name__)

logger.setLevel(level="INFO")


def isprior(obj):
    if isinstance(obj, af.Model):
        return True
    return False


def isinstance_or_prior(obj, cls):
    if isinstance(obj, cls):
        return True
    if isinstance(obj, af.Model) and obj.cls == cls:
        return True
    return False


def mesh_list_from(model: af.Collection) -> List[aa.AbstractMesh]:
    """
    For a model containing one or more galaxies, inspect its attributes and return the list of `mesh`'s of each
    `pixelization` of all galaxies. If no galaxy has pixelization an empty list is returned.

    This function expects that the input model is a `Collection` where the first model-component has the
    name `galaxies`, and is itself a `Collection` of `Galaxy` instances. This is the
    standard API for creating a model in PyAutoGalaxy.

    The result of `mesh_from` is used by the preloading to determine whether certain parts of a
    calculation can be cached before the non-linear search begins for efficiency.

    Parameters
    ----------
    model
        Contains the `galaxies` in the model that will be fitted via the non-linear search.

    Returns
    -------
    The `mesh` of a galaxy, provided one galaxy has a `mesh`.
    """

    instance = model.instance_from_prior_medians(ignore_prior_limits=True)

    mesh_list = []

    for galaxy in instance.galaxies:

        pixelization_list = galaxy.cls_list_from(cls=aa.Pixelization)

        for pixelization in pixelization_list:
            if pixelization is not None:
                mesh_list.append(pixelization.mesh)

    return mesh_list


def has_pixelization_from(model: af.Collection) -> bool:
    """
    For a model containing one or more galaxies, inspect its attributes and return `True` if a galaxy has a
    `Pixelization` otherwise return `False`.

    This function expects that the input model is a `Collection` where the first model-component has the
    name `galaxies`, and is itself a `Collection` of `Galaxy` instances. This is the
    standard API for creating a model in PyAutoGalaxy.

    The result of `has_pixelization_from` is used by the preloading to determine whether certain parts of a
    calculation can be cached before the non-linear search begins for efficiency.

    Parameters
    ----------
    model : af.Collection
        Contains the `galaxies` in the model that will be fitted via the non-linear search.

    Returns
    -------
    aa.mesh.Mesh or None:
        The `Pixelization` of a galaxy, provided one galaxy has a `Pixelization`.
    """
    mesh_list = mesh_list_from(model=model)

    return len(mesh_list) > 0


def set_upper_limit_of_pixelization_pixels_prior(
    model: af.Collection,
    pixels_in_mask: int,
    lower_limit_no_pixels_below_mask: int = 10,
):
    """
    If the mesh(es) of pixelizations being fitted in the hyper-model fit is a `VoronoiBrightnessImage` pixelization,
    this function sets the upper limit of its `pixels` prior to the number of data points in the mask.

    This ensures the KMeans algorithm does not raise an exception due to having fewer data points than source pixels.

    Parameters
    ----------
    model
        The hyper model used by the hyper-fit, which models hyper-components like a `Pixelization` or `HyperGalaxy`'s.
    pixels_in_mask
        The number of pixels in the mask, which are used to set the upper and lower limits of the priors on the
        number of pixels in the pixelization.
    lower_limit_no_pixels_below_mask
        If the prior lower limit on the pixelization's number of pixels is above the number of pixels in the mask,
        the number of pixels in the mask below which the lower limit is set.
    """

    if not hasattr(model, "galaxies"):
        return

    mesh_list = model.galaxies.models_with_type(
        cls=(
            aa.mesh.DelaunayBrightnessImage,
            aa.mesh.VoronoiBrightnessImage,
            aa.mesh.VoronoiNNBrightnessImage,
        )
    )

    if not mesh_list:
        return

    for mesh in mesh_list:

        if hasattr(mesh.pixels, "upper_limit"):

            if pixels_in_mask < mesh.pixels.upper_limit:

                lower_limit = mesh.pixels.lower_limit

                log_str = (
                    "MODIFY BEFORE FIT -  A pixelization mesh's pixel UniformPrior upper limit"
                    "was greater than the number of pixels in the mask. It has been "
                    "reduced to the number of pixels in the mask.\,"
                )

                if lower_limit > pixels_in_mask:

                    lower_limit = pixels_in_mask - lower_limit_no_pixels_below_mask

                    logger.info(
                        log_str
                        + "MODIFY BEFORE FIT - The pixelization's mesh's pixel UniformPrior lower_limit was "
                        "also above the number of pixels in the mask, and has been reduced"
                        "to the number of pixels in the mask minus 10."
                    )
                else:
                    logger.info(log_str)

                mesh.pixels = af.UniformPrior(
                    lower_limit=lower_limit,
                    upper_limit=pixels_in_mask,
                )


def clean_model_of_hyper_images(model):

    for galaxy in model.galaxies:

        del galaxy.hyper_model_image
        del galaxy.hyper_galaxy_image

    if hasattr(model, "clumps"):
        for clump in model.clumps:

            del clump.hyper_model_image
            del clump.hyper_galaxy_image

    return model


def hyper_noise_model_from(
    setup_hyper, result: af.Result, include_hyper_image_sky: bool = False
) -> af.Collection:
    """
    Make a hyper model from the `Result` of a model-fit, where the hyper-model is the maximum log likelihood instance
    of the inferred model but turns the following hyper components of the model to free parameters:

    1) The `Pixelization` of any `Galaxy` in the model.
    2) The `Regularization` of any `Galaxy` in the model.
    3) Hyper data components like a `HyperImageSky` or `HyperBackgroundNoise` if input into the function.
    4) `HyperGalaxy` components of the `Galaxy`'s in the model, which are used to scale the noise in regions of the
       data which are fit poorly.

    The hyper model is typically used in pipelines to refine and improve an `LEq` after model-fits that fit the
    `Galaxy` light and mass components.

    Parameters
    ----------
    setup_hyper
        The setup of the hyper analysis if used (e.g. hyper-galaxy noise scaling).
    result
        The result of a previous `Analysis` search whose maximum log likelihood model forms the basis of the hyper model.
    include_hyper_image_sky
        This must be true to include the hyper-image sky in the model, even if it is turned on in `setup_hyper`.

    Returns
    -------
    af.Collection
        The hyper model, which has an instance of the input results maximum log likelihood model with certain hyper
        model components now free parameters.
    """

    from autogalaxy.galaxy.hyper import HyperGalaxy

    if setup_hyper is None:
        return None

    if setup_hyper.hyper_galaxy_names is None:
        if setup_hyper.hypers_all_off:
            return None
        if setup_hyper.hypers_all_except_image_sky_off:
            if not include_hyper_image_sky:
                return None

    model = result.instance.as_model()

    model = clean_model_of_hyper_images(model=model)

    model.hyper_image_sky = setup_hyper.hyper_image_sky
    model.hyper_background_noise = setup_hyper.hyper_background_noise

    if setup_hyper.hyper_galaxy_names is not None:

        for path_galaxy, galaxy in result.path_galaxy_tuples:
            if path_galaxy[-1] in setup_hyper.hyper_galaxy_names:
                if not np.all(result.hyper_galaxy_image_path_dict[path_galaxy] == 0):

                    galaxy = getattr(model.galaxies, path_galaxy[-1])

                    setattr(galaxy, "hyper_galaxy", af.Model(HyperGalaxy))

    return model


def hyper_pix_model_from(
    setup_hyper,
    result: af.Result,
    include_hyper_image_sky: bool = False,
    pixelization_overwrite=None,
    regularization_overwrite=None,
) -> af.Collection:
    """
    Make a hyper model from the `Result` of a model-fit, where the hyper-model is the maximum log likelihood instance
    of the inferred model but turns the following hyper components of the model to free parameters:

    1) The `Pixelization` of any `Galaxy` in the model.
    2) The `Regularization` of any `Galaxy` in the model.
    3) Hyper data components like a `HyperImageSky` or `HyperBackgroundNoise` if input into the function.
    4) `HyperGalaxy` components of the `Galaxy`'s in the model, which are used to scale the noise in regions of the
       data which are fit poorly.

    The hyper model is typically used in pipelines to refine and improve an `LEq` after model-fits that fit the
    `Galaxy` light and mass components.

    Parameters
    ----------
    setup_hyper
        The setup of the hyper analysis if used (e.g. hyper-galaxy noise scaling).
    result
        The result of a previous `Analysis` search whose maximum log likelihood model forms the basis of the hyper model.
    include_hyper_image_sky
        This must be true to include the hyper-image sky in the model, even if it is turned on in `setup_hyper`.

    Returns
    -------
    af.Collection
        The hyper model, which has an instance of the input results maximum log likelihood model with certain hyper
        model components now free parameters.
    """

    if setup_hyper is None:
        return None

    model = result.instance.as_model(
        model_classes=(aa.AbstractMesh, aa.AbstractRegularization),
        excluded_classes=(aa.reg.ConstantZeroth, aa.reg.Zeroth),
    )

    if not has_pixelization_from(model=model):
        return None

    if pixelization_overwrite:
        model.galaxies.source.pixelization = af.Model(pixelization_overwrite)

    if regularization_overwrite:
        model.galaxies.source.regularization = af.Model(regularization_overwrite)

    if setup_hyper.mesh_pixels_fixed is not None:
        if hasattr(model.galaxies.source.pixelization.mesh, "pixels"):
            model.galaxies.source.pixelization.mesh.pixels = (
                setup_hyper.mesh_pixels_fixed
            )

    model = clean_model_of_hyper_images(model=model)

    model.hyper_image_sky = None
    model.hyper_background_noise = None

    if setup_hyper.hyper_image_sky is not None and include_hyper_image_sky:
        model.hyper_image_sky = setup_hyper.hyper_image_sky

    if setup_hyper.hyper_background_noise is not None:
        try:
            model.hyper_background_noise = result.instance.hyper_background_noise
        except AttributeError:
            pass

    if setup_hyper.hyper_galaxy_names is not None:

        for path_galaxy, galaxy in result.path_galaxy_tuples:
            if path_galaxy[-1] in setup_hyper.hyper_galaxy_names:
                if not np.all(result.hyper_galaxy_image_path_dict[path_galaxy] == 0):

                    model_galaxy = getattr(model.galaxies, path_galaxy[-1])

                    setattr(model_galaxy, "hyper_galaxy", galaxy.hyper_galaxy)

    return model


def hyper_fit_no_noise(
    setup_hyper,
    result: af.Result,
    analysis,
    search_previous,
    use_positive_only_solver: bool = False,
    include_hyper_image_sky: bool = False,
):

    analysis.set_hyper_dataset(result=result)

    if use_positive_only_solver:

        analysis.settings_inversion.use_positive_only_solver = True

    hyper_model_pix = hyper_pix_model_from(
        setup_hyper=setup_hyper,
        result=result,
        include_hyper_image_sky=include_hyper_image_sky,
    )

    if hyper_model_pix is None:

        return result

    search = setup_hyper.search_pix_cls(
        path_prefix=search_previous.path_prefix_no_unique_tag,
        name=f"{search_previous.paths.name}__hyper_pix",
        unique_tag=search_previous.paths.unique_tag,
        number_of_cores=search_previous.number_of_cores,
        **setup_hyper.search_pix_dict,
    )

    hyper_pix_result = search.fit(model=hyper_model_pix, analysis=analysis)

    result.hyper = hyper_pix_result

    return result


def hyper_fit(
    setup_hyper,
    result: af.Result,
    analysis,
    search_previous: af.NonLinearSearch,
    use_positive_only_solver: bool = False,
    include_hyper_image_sky: bool = False,
    pixelization_overwrite=None,
    regularization_overwrite=None,
):
    """
    Perform a hyper-fit, which extends a model-fit with an additional fit which fixes the non-hyper components of the
    model (e.g., `LightProfile`'s, `MassProfile`) to the `Result`'s maximum likelihood fit. The hyper-fit then treats
    only the hyper-model components as free parameters, which are any of the following model components:

    1) The `Pixelization` of any `Galaxy` in the model.
    2) The `Regularization` of any `Galaxy` in the model.
    3) Hyper data components like a `HyperImageSky` or `HyperBackgroundNoise` if input into the function.
    4) `HyperGalaxy` components of the `Galaxy`'s in the model, which are used to scale the noise in regions of the
       data which are fit poorly.

    The hyper model is typically used in pipelines to refine and improve an `LEq` after model-fits that fit the
    `Galaxy` light and mass components.

    Parameters
    ----------
    hyper_model : Collection
        The hyper model used by the hyper-fit, which models hyper-components like a `Pixelization` or `HyperGalaxy`'s.
    setup_hyper : SetupHyper
        The setup of the hyper analysis if used (e.g. hyper-galaxy noise scaling).
    result : af.Result
        The result of a previous `Analysis` search whose maximum log likelihood model forms the basis of the hyper model.
    analysis : Analysis
        An analysis class used to fit imaging or interferometer data with a model.
    use_positive_only_solver
        Use a positive-only solver for the linear algebra, which does not allow negative values in the lens light /
        source reconstructions.

    Returns
    -------
    af.Result
        The result of the hyper model-fit, which has a new attribute `result.hyper` that contains updated parameter
        values for the hyper-model components for passing to later model-fits.
    """

    if analysis.hyper_model_image is None:

        analysis.set_hyper_dataset(result=result)

    hyper_noise_model = hyper_noise_model_from(
        setup_hyper=setup_hyper,
        result=result,
        include_hyper_image_sky=include_hyper_image_sky,
    )

    if hyper_noise_model is None:

        return hyper_fit_no_noise(
            setup_hyper=setup_hyper,
            result=result,
            analysis=analysis,
            search_previous=search_previous,
            use_positive_only_solver=use_positive_only_solver,
            include_hyper_image_sky=include_hyper_image_sky,
        )

    if use_positive_only_solver:
        analysis.settings_inversion.use_positive_only_solver = True

    if hyper_noise_model is not None:

        search = setup_hyper.search_noise_cls(
            path_prefix=search_previous.path_prefix_no_unique_tag,
            name=f"{search_previous.paths.name}__hyper_noise",
            unique_tag=search_previous.paths.unique_tag,
            number_of_cores=search_previous.number_of_cores,
            **setup_hyper.search_noise_dict,
        )

        hyper_noise_result = search.fit(model=hyper_noise_model, analysis=analysis)

    analysis.set_hyper_dataset(result=result)

    hyper_pix_model = hyper_pix_model_from(
        setup_hyper=setup_hyper,
        result=hyper_noise_result,
        include_hyper_image_sky=include_hyper_image_sky,
        pixelization_overwrite=pixelization_overwrite,
        regularization_overwrite=regularization_overwrite,
    )

    if hyper_pix_model is None:

        result.hyper = hyper_noise_result

        return result

    try:
        set_upper_limit_of_pixelization_pixels_prior(
            model=hyper_pix_model, pixels_in_mask=result.mask.pixels_in_mask
        )
    except AttributeError:
        pass

    search = setup_hyper.search_pix_cls(
        path_prefix=search_previous.path_prefix_no_unique_tag,
        name=f"{search_previous.paths.name}__hyper_pix",
        unique_tag=search_previous.paths.unique_tag,
        number_of_cores=search_previous.number_of_cores,
        **setup_hyper.search_pix_dict,
    )

    hyper_pix_result = search.fit(model=hyper_pix_model, analysis=analysis)

    result.hyper = hyper_pix_result

    return result


def hyper_model_from(
    setup_hyper, result: af.Result, include_hyper_image_sky: bool = False
) -> af.Collection:
    """
    Make a hyper model from the `Result` of a model-fit, where the hyper-model is the maximum log likelihood instance
    of the inferred model but turns the following hyper components of the model to free parameters:
    1) The `Pixelization` of any `Galaxy` in the model.
    2) The `Regularization` of any `Galaxy` in the model.
    3) Hyper data components like a `HyperImageSky` or `HyperBackgroundNoise` if input into the function.
    4) `HyperGalaxy` components of the `Galaxy`'s in the model, which are used to scale the noise in regions of the
       data which are fit poorly.

    The hyper model is typically used in pipelines to refine and improve an `LEq` after model-fits that fit the
    `Galaxy` light and mass components.

    Parameters
    ----------
    setup_hyper
        The setup of the hyper analysis if used (e.g. hyper-galaxy noise scaling).
    result
        The result of a previous `Analysis` search whose maximum log likelihood model forms the basis of the hyper model.
    include_hyper_image_sky
        This must be true to include the hyper-image sky in the model, even if it is turned on in `setup_hyper`.
    Returns
    -------
    af.Collection
        The hyper model, which has an instance of the input results maximum log likelihood model with certain hyper
        model components now free parameters.
    """

    from autogalaxy.galaxy.hyper import HyperGalaxy

    model = result.instance.as_model(
        model_classes=(aa.AbstractMesh, aa.AbstractRegularization),
        excluded_classes=(aa.reg.ConstantZeroth, aa.reg.Zeroth),
    )

    model = clean_model_of_hyper_images(model=model)

    if setup_hyper is None:
        return None

    if setup_hyper.hyper_galaxy_names is None:
        if not has_pixelization_from(model=model):
            if setup_hyper.hypers_all_off:
                return None
            if setup_hyper.hypers_all_except_image_sky_off:
                if not include_hyper_image_sky:
                    return None

    model.hyper_image_sky = setup_hyper.hyper_image_sky
    model.hyper_background_noise = setup_hyper.hyper_background_noise

    if setup_hyper.hyper_galaxy_names is not None:

        for path_galaxy, galaxy in result.path_galaxy_tuples:
            if path_galaxy[-1] in setup_hyper.hyper_galaxy_names:
                if not np.all(result.hyper_galaxy_image_path_dict[path_galaxy] == 0):

                    galaxy = getattr(model.galaxies, path_galaxy[-1])

                    setattr(galaxy, "hyper_galaxy", af.Model(HyperGalaxy))

    return model


def stochastic_model_from(
    result,
    include_lens_light=False,
    include_pixelization=False,
    include_regularization=False,
    subhalo_centre_width=None,
    subhalo_mass_at_200_log_uniform=True,
    clean_model: bool = True,
):
    """
    Make a stochastic model from  the `Result` of a model-fit, where the stochastic model uses the same model
    components as the original model but may switch certain components (e.g. the lens light, source pixelization)
    to free parameters.

    The stochastic model is used to perform a stochastic model-fit, which refits a model but introduces a log
    likelihood cap whereby all model-samples with a likelihood above this cap are rounded down to the value of the cap.

    This `log_likelihood_cap` is determined by sampling ~250 log likeilhood values from the original model's, but where
    each model evaluation uses a different KMeans seed of the pixelization to derive a unique pixelization with which
    to reconstruct the source galaxy (therefore a pixelization which uses the KMeans method, like the
    `VoronoiBrightnessImage` must be used to perform a stochastic fit).

    The cap is computed as the mean of these ~250 values and it is introduced o avoid underestimated errors due
    to artificial likelihood boosts.

    Parameters
    ----------
    result : af.Result
        The result of a previous `Analysis` search whose maximum log likelihood model forms the basis of the hyper model.
    include_lens_light
        If `True` and the model includes any `LightProfile`'s, these are fitted for in the model.
    include_pixelization
        If `True` the `VoronoiBrightnessImage` pixelization in the model is fitted for.
    include_regularization
        If `True` the regularization in the model is fitted for.
    subhalo_centre_width
        The `sigma` value of the `GaussianPrior` on the centre of the subhalo, if it is included in the lens model.
    subhalo_mass_at_200_log_uniform
        if `True`, the subhalo mass (if included) does not assume a `GaussianPrior` from the previous fit, but instead
        retains the default `LogUniformPrior`.

    Returns
    -------
    af.Collection
        The stochastic model, which is the same model as the input model but may fit for or fix additional parameters.
    """
    if not hasattr(result.model.galaxies, "lens"):
        raise exc.PriorException(
            "Cannot extend a search with a stochastic search if the lens galaxy `Model` "
            "is not named `lens`. "
        )

    model_classes = [MassProfile]

    if include_lens_light:
        model_classes.append(LightProfile)

    if include_pixelization:
        model_classes.append(aa.AbstractMesh)

    if include_regularization:
        model_classes.append(aa.AbstractRegularization)

    model = result.instance.as_model(tuple(model_classes))

    if clean_model:
        model = clean_model_of_hyper_images(model=model)

    model.galaxies.lens.take_attributes(source=result.model.galaxies.lens)

    if hasattr(model, "clumps"):
        model.clumps = result.model.clumps

    if hasattr(model.galaxies, "subhalo"):
        model.galaxies.subhalo.take_attributes(source=result.model.galaxies.subhalo)

        if subhalo_centre_width is not None:
            model.galaxies.subhalo.mass.centre = result.model_absolute(
                a=subhalo_centre_width
            ).galaxies.subhalo.mass.centre

        if subhalo_mass_at_200_log_uniform:
            model.galaxies.subhalo.mass.mass_at_200 = af.LogUniformPrior(
                lower_limit=1e6, upper_limit=1e12
            )

    return model


def stochastic_fit(
    stochastic_model: af.Collection,
    search_cls: ClassVar[af.NonLinearSearch],
    search_pix_dict: Dict,
    result: ResultDataset,
    analysis: AnalysisDataset,
    search_previous: af.NonLinearSearch,
    info: Dict = None,
    pickle_files: List[str] = None,
):
    """
    Perform a stochastic model-fit, which refits a model but introduces a log likelihood cap whereby all model-samples
    with a likelihood above this cap are rounded down to the value of the cap.

    This `log_likelihood_cap` is determined by sampling ~250 log likelihood values from the original model's maximum
    log likelihood model. However, the pixelization used to reconstruct the source of each model evaluation uses a
    different KMeans seed, such that each reconstruction uses a unique pixel-grid. The model must therefore use a
    pixelization which uses the KMeans method to construct the pixel-grid, for example the `VoronoiBrightnessImage`.

    The cap is computed as the mean of these ~250 values and it is introduced to avoid underestimated errors due
    to artificial likelihood boosts.

    Parameters
    ----------
    setup_hyper : SetupHyper
        The setup of the hyper analysis if used (e.g. hyper-galaxy noise scaling).
    result : af.Result
        The result of a previous `Analysis` search whose maximum log likelihood model forms the basis of the hyper model.
    include_hyper_image_sky : hd.HyperImageSky
        This must be true to include the hyper-image sky in the model, even if it is turned on in `setup_hyper`.

    Returns
    -------
    af.Collection
        The hyper model, which has an instance of the input results maximum log likelihood model with certain hyper
        model components now free parameters.
    """

    mean, sigma = norm.fit(
        result.stochastic_log_likelihoods_from(paths=search_previous.paths)
    )
    log_likelihood_cap = mean

    name = f"{search_previous.paths.name}__stochastic"

    search = search_cls(
        path_prefix=search_previous.path_prefix_no_unique_tag,
        name=name,
        unique_tag=search_previous.paths.unique_tag,
        number_of_cores=search_previous.number_of_cores,
        **search_pix_dict,
    )

    stochastic_result = search.fit(
        model=stochastic_model,
        analysis=analysis,
        log_likelihood_cap=log_likelihood_cap,
        info=info,
        pickle_files=pickle_files,
    )

    search.paths.restore()

    search.paths.save_object(
        "stochastic_log_likelihoods",
        result.stochastic_log_likelihoods_from(paths=search_previous.paths),
    )

    search.paths.zip_remove()

    result.stochastic = stochastic_result

    return result
