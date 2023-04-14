from os import path

import autofit as af
import autogalaxy as ag

directory = path.dirname(path.realpath(__file__))


def test__figure_of_merit__includes_hyper_image_and_noise__matches_fit(
    masked_imaging_7x7,
):

    hyper_image_sky = ag.legacy.hyper_data.HyperImageSky(sky_scale=1.0)
    hyper_background_noise = ag.legacy.hyper_data.HyperBackgroundNoise(noise_scale=1.0)

    galaxy = ag.legacy.Galaxy(redshift=0.5, light=ag.lp.Sersic(intensity=0.1))

    model = af.Collection(
        hyper_image_sky=hyper_image_sky,
        hyper_background_noise=hyper_background_noise,
        galaxies=af.Collection(galaxy=galaxy),
    )

    analysis = ag.legacy.AnalysisImaging(dataset=masked_imaging_7x7)

    instance = model.instance_from_unit_vector([])
    fit_figure_of_merit = analysis.log_likelihood_function(instance=instance)

    plane = analysis.plane_via_instance_from(instance=instance)
    fit = ag.legacy.FitImaging(
        dataset=masked_imaging_7x7,
        plane=plane,
        hyper_image_sky=hyper_image_sky,
        hyper_background_noise=hyper_background_noise,
    )

    assert fit.log_likelihood == fit_figure_of_merit


def test__uses_hyper_fit_correctly(masked_imaging_7x7):

    galaxies = af.ModelInstance()
    galaxies.galaxy = ag.legacy.Galaxy(
        redshift=0.5, light=ag.lp.Sersic(intensity=1.0), mass=ag.mp.IsothermalSph
    )
    galaxies.source = ag.legacy.Galaxy(redshift=1.0, light=ag.lp.Sersic())

    instance = af.ModelInstance()
    instance.galaxies = galaxies

    galaxy_hyper_image = ag.Array2D.ones(shape_native=(3, 3), pixel_scales=0.1)
    galaxy_hyper_image[4] = 10.0
    hyper_model_image = ag.Array2D.full(
        fill_value=0.5, shape_native=(3, 3), pixel_scales=0.1
    )

    hyper_galaxy_image_path_dict = {("galaxies", "galaxy"): galaxy_hyper_image}

    result = ag.m.MockResult(
        hyper_galaxy_image_path_dict=hyper_galaxy_image_path_dict,
        hyper_model_image=hyper_model_image,
    )

    analysis = ag.legacy.AnalysisImaging(
        dataset=masked_imaging_7x7, hyper_dataset_result=result
    )

    hyper_galaxy = ag.legacy.HyperGalaxy(
        contribution_factor=1.0, noise_factor=1.0, noise_power=1.0
    )

    instance.galaxies.galaxy.hyper_galaxy = hyper_galaxy

    fit_likelihood = analysis.log_likelihood_function(instance=instance)

    g0 = ag.legacy.Galaxy(
        redshift=0.5,
        light_profile=instance.galaxies.galaxy.light,
        mass_profile=instance.galaxies.galaxy.mass,
        hyper_galaxy=hyper_galaxy,
        hyper_model_image=hyper_model_image,
        hyper_galaxy_image=galaxy_hyper_image,
        hyper_minimum_value=0.0,
    )
    g1 = ag.legacy.Galaxy(redshift=1.0, light_profile=instance.galaxies.source.light)

    plane = ag.legacy.Plane(galaxies=[g0, g1])

    fit = ag.legacy.FitImaging(dataset=masked_imaging_7x7, plane=plane)

    assert (fit.plane.galaxies[0].hyper_galaxy_image == galaxy_hyper_image).all()
    assert fit_likelihood == fit.log_likelihood
