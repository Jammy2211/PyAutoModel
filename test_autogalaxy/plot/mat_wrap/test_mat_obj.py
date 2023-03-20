import autogalaxy.plot as aplt


def test__mat_obj__all_load_from_config_correctly():

    light_profile_centres_scatter = aplt.LightProfileCentresScatter()

    assert light_profile_centres_scatter.config_dict["s"] == 1

    mass_profile_centres_scatter = aplt.MassProfileCentresScatter()

    assert mass_profile_centres_scatter.config_dict["s"] == 2

    multiple_images_scatter = aplt.MultipleImagesScatter()

    assert multiple_images_scatter.config_dict["s"] == 3

    tangential_critical_curves_plot = aplt.TangentialCriticalCurvesPlot()

    assert tangential_critical_curves_plot.config_dict["width"] == 4

    tangential_caustics_plot = aplt.TangentialCausticsPlot()

    assert tangential_caustics_plot.config_dict["width"] == 5

    radial_critical_curves_plot = aplt.RadialCriticalCurvesPlot()

    assert radial_critical_curves_plot.config_dict["width"] == 4

    radial_caustics_plot = aplt.RadialCausticsPlot()

    assert radial_caustics_plot.config_dict["width"] == 5
