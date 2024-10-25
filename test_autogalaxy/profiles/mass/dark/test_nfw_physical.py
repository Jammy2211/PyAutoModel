import numpy as np
import pytest

import autofit as af
import autogalaxy as ag


def test_nfw_to_from():
    cosmo = ag.cosmology.Planck15()
    zhalo = 0.3
    zsrc = 1.0

    model = af.Model(ag.mp.NFWPhysical, redshift_object=zhalo, redshift_source=zsrc, mdef='200c', cosmo=cosmo)
    model.concentration = af.UniformPrior(lower_limit=3.0, upper_limit=15.0)

    # we convert to and from physical NFW and check the (mass, concentration) is the same
    for i in range(10):
        instance = model.random_instance()
        true_mass = instance.params['log10M']
        calc_mass = np.log10(instance.mass_at_200_solar_masses(
            zhalo, zsrc, cosmology=cosmo
        ))
        assert calc_mass == pytest.approx(true_mass, 1e-2)
        true_c = instance.params['concentration']
        calc_c = instance.concentration(
            zhalo, zsrc, cosmology=cosmo
        )
        assert calc_c == pytest.approx(true_c, 1e-2)

    zhalo = 0.5
    zsrc = 2.0

    model = af.Model(ag.mp.NFWPhysical, redshift_object=zhalo, redshift_source=zsrc, cosmo=cosmo)
    model.concentration = af.UniformPrior(lower_limit=3.0, upper_limit=15.0)

    # we convert to and from physical NFW and check the (mass, concentration) is the same
    for i in range(10):
        instance = model.random_instance()
        true_mass = instance.params['log10M']
        calc_mass = np.log10(instance.mass_at_200_solar_masses(
            zhalo, zsrc, cosmology=cosmo
        ))
        assert calc_mass == pytest.approx(true_mass, 1e-2)
        true_c = instance.params['concentration']
        calc_c = instance.concentration(
            zhalo, zsrc, cosmology=cosmo
        )
        assert calc_c == pytest.approx(true_c, 1e-2)


def test_other_mass_defs():
    cosmo = ag.cosmology.Planck15()
    zhalo = 0.3
    zsrc = 1.0

    model = af.Model(ag.mp.NFWPhysical, redshift_object=zhalo, redshift_source=zsrc, mdef='vir', cosmo=cosmo)
    model.concentration = af.UniformPrior(lower_limit=3.0, upper_limit=15.0)

    for i in range(10):
        instance = model.random_instance()
        true_mass = instance.params['log10M']
        calc_mass = np.log10(instance.mass_at_200_solar_masses(
            zhalo, zsrc, cosmology=cosmo
        ))
        true_c = instance.params['concentration']
        calc_c = instance.concentration(
            zhalo, zsrc, cosmology=cosmo
        )

    model = af.Model(ag.mp.NFWPhysical, redshift_object=zhalo, redshift_source=zsrc, mdef='500m', cosmo=cosmo)
    model.concentration = af.UniformPrior(lower_limit=3.0, upper_limit=15.0)

    for i in range(10):
        instance = model.random_instance()
        true_mass = instance.params['log10M']
        calc_mass = np.log10(instance.mass_at_200_solar_masses(
            zhalo, zsrc, cosmology=cosmo
        ))
        true_c = instance.params['concentration']
        calc_c = instance.concentration(
            zhalo, zsrc, cosmology=cosmo
        )
