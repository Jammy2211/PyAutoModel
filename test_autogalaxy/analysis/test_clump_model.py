import autofit as af
import autogalaxy as ag


def test__clumps():
    centres = ag.Grid2DIrregular(values=[(1.0, 1.0)])

    clump_model = ag.ClumpModel(
        redshift=0.5, centres=centres, light_cls=ag.lp.SersicSph
    )

    clumps = clump_model.clumps

    assert clumps["clump_0"].redshift == 0.5
    assert clumps["clump_0"].light.centre == (1.0, 1.0)
    assert isinstance(clumps["clump_0"].light.intensity, af.LogUniformPrior)
    assert clumps["clump_0"].mass == None

    clump_model = ag.ClumpModel(
        redshift=0.5, centres=centres, mass_cls=ag.mp.IsothermalSph
    )

    clumps = clump_model.clumps

    assert clumps["clump_0"].redshift == 0.5
    assert clumps["clump_0"].mass.centre == (1.0, 1.0)
    assert isinstance(clumps["clump_0"].mass.einstein_radius, af.UniformPrior)
    assert clumps["clump_0"].light == None

    clump_model = ag.ClumpModel(
        redshift=0.5,
        centres=centres,
        light_cls=ag.lp.SersicSph,
        mass_cls=ag.mp.IsothermalSph,
    )

    clumps = clump_model.clumps

    assert clumps["clump_0"].redshift == 0.5
    assert clumps["clump_0"].light.centre == (1.0, 1.0)
    assert clumps["clump_0"].mass.centre == (1.0, 1.0)
    assert isinstance(clumps["clump_0"].light.intensity, af.LogUniformPrior)
    assert isinstance(clumps["clump_0"].mass.einstein_radius, af.UniformPrior)


def test__clumps_light_only():
    centres = ag.Grid2DIrregular(values=[(1.0, 1.0)])

    clump_model = ag.ClumpModel(
        redshift=0.5,
        centres=centres,
        light_cls=ag.lp.SersicSph,
        mass_cls=ag.mp.IsothermalSph,
    )

    clumps_light_only = clump_model.clumps_light_only

    assert clumps_light_only["clump_0"].redshift == 0.5
    assert clumps_light_only["clump_0"].light.centre == (1.0, 1.0)
    assert isinstance(clumps_light_only["clump_0"].light.intensity, af.LogUniformPrior)
    assert not hasattr(clumps_light_only["clump_0"], "mass")


def test__clumps_mass_only():
    centres = ag.Grid2DIrregular(values=[(1.0, 1.0)])

    clump_model = ag.ClumpModel(
        redshift=0.5,
        centres=centres,
        light_cls=ag.lp.SersicSph,
        mass_cls=ag.mp.IsothermalSph,
    )

    clumps_mass_only = clump_model.clumps_mass_only

    assert clumps_mass_only["clump_0"].redshift == 0.5
    assert clumps_mass_only["clump_0"].mass.centre == (1.0, 1.0)
    assert isinstance(clumps_mass_only["clump_0"].mass.einstein_radius, af.UniformPrior)
    assert not hasattr(clumps_mass_only["clump_0"], "light")


def test__einstein_radius_max():
    centres = ag.Grid2DIrregular(values=[(1.0, 1.0)])

    clump_model = ag.ClumpModel(
        redshift=0.5,
        centres=centres,
        mass_cls=ag.mp.IsothermalSph,
        einstein_radius_upper_limit=1.0,
    )

    clumps = clump_model.clumps

    assert isinstance(clumps["clump_0"].mass.einstein_radius, af.UniformPrior)
    assert clumps["clump_0"].mass.einstein_radius.upper_limit == 1.0

    assert clumps["clump_0"].redshift == 0.5
    assert clumps["clump_0"].mass.centre == (1.0, 1.0)
    assert clumps["clump_0"].light == None
