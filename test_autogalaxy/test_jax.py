import jax
import pytest

import autoarray as aa
import autofit as af
import autogalaxy as ag


@pytest.fixture(name="instance")
def make_instance():
    galaxy_model = af.Model(ag.Galaxy, redshift=0.5, bulge=ag.lp_operated.Sersic)
    model = af.Collection(galaxies=af.Collection(galaxy=galaxy_model))
    return model.instance_from_prior_medians()


@pytest.fixture(name="analysis")
def make_analysis():
    imaging = ag.Imaging(
        image=aa.Array2D.full(0.0, shape_native=(10, 10), pixel_scales=0.05),
        noise_map=aa.Array2D.full(1.0, shape_native=(10, 10), pixel_scales=0.05),
        psf=aa.Kernel2D.ones(shape_native=(3, 3), pixel_scales=0.05, normalize=True),
    )

    mask_2d = ag.Mask2D.circular(
        shape_native=imaging.shape_native, pixel_scales=imaging.pixel_scales, radius=3.0
    )

    imaging = imaging.apply_mask(mask=mask_2d)

    return ag.AnalysisImaging(dataset=imaging)


def test_gradient(analysis, instance):
    gradient = jax.grad(analysis.log_likelihood_function)
    gradient_instance = gradient(instance)


def test_jit(analysis, instance):
    jitted = jax.jit(analysis.log_likelihood_function)
    called = jitted(instance)
