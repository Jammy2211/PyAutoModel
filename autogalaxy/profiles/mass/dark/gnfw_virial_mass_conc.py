from typing import Tuple

from autogalaxy.profiles.mass.dark.gnfw import gNFWSph

from astropy import units

import numpy as np
from autogalaxy import cosmology as cosmo

from scipy.integrate import quad


def kappa_s_and_scale_radius(
    cosmology, virial_mass, c_2, overdens, redshift_object, redshift_source, inner_slope
):
    concentration = (2 - inner_slope) * c_2  # gNFW concentration

    critical_density = (
        cosmology.critical_density(redshift_object).to(units.solMass / units.kpc**3)
    ).value

    critical_surface_density = (
        cosmology.critical_surface_density_between_redshifts_solar_mass_per_kpc2_from(
            redshift_0=redshift_object, redshift_1=redshift_source
        )
    )

    kpc_per_arcsec = cosmology.kpc_per_arcsec_from(redshift=redshift_object)

    if overdens == 0:
        x = cosmology.Om(redshift_object) - 1
        overdens = 18 * np.pi**2 + 82 * x - 39 * x**2  # Bryan & Norman (1998)

    virial_radius = (
        virial_mass / (overdens * critical_density * (4.0 * np.pi / 3.0))
    ) ** (
        1.0 / 3.0
    )  # r_vir

    scale_radius_kpc = (
        virial_radius / concentration
    )  # scale radius of gNFW profile in kpc

    ##############################
    def integrand(r):
        return (r**2 / r**inner_slope) * (1 + r / scale_radius_kpc) ** (inner_slope - 3)

    de_c = (
        (overdens / 3.0)
        * (virial_radius**3 / scale_radius_kpc**inner_slope)
        / quad(integrand, 0, virial_radius)[0]
    )  # rho_c
    ##############################

    rho_s = critical_density * de_c  # rho_s
    kappa_s = rho_s * scale_radius_kpc / critical_surface_density  # kappa_s
    scale_radius = scale_radius_kpc / kpc_per_arcsec  # scale radius in arcsec

    return kappa_s, scale_radius, virial_radius, overdens


class gNFWVirialMassConcSph(gNFWSph):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        log10m_vir: float = 12.0,
        c_2: float = 10.0,
        overdens: float = 0.0,
        redshift_object: float = 0.5,
        redshift_source: float = 1.0,
        inner_slope: float = 1.0,
    ):
        """
        Spherical gNFW profile initialized with the virial mass and c_2 concentration of the halo.

        The virial radius of the halo is defined as the radius at which the density of the halo
        equals overdens * the critical density of the Universe. r_vir = (3*m_vir/4*pi*overdens*critical_density)^1/3.

        If the overdens parameter is set to 0, the virial overdensity of Bryan & Norman (1998) will be used.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        log10m_vir
            The log10(virial mass) of the dark matter halo.
        c_2
            The c_2 concentration of the dark matter halo, which equals r_vir/r_2, where r_2 is the
            radius at which the logarithmic density slope equals -2.
        overdens
            The spherical overdensity used to define the virial radius of the dark matter
            halo: r_vir = (3*m_vir/4*pi*overdens*critical_density)^1/3. If this parameter is set to 0, the virial
            overdensity of Bryan & Norman (1998) will be used.
        redshift_object
            Lens redshift.
        redshift_source
            Source redshift.
        inner_slope
            The inner slope of the dark matter halo's gNFW density profile.
        """

        self.log10m_vir = log10m_vir
        self.c_2 = c_2
        self.redshift_object = redshift_object
        self.redshift_source = redshift_source
        self.inner_slope = inner_slope

        (
            kappa_s,
            scale_radius,
            virial_radius,
            overdens,
        ) = kappa_s_and_scale_radius(
            cosmology=cosmo.Planck15(),
            virial_mass=10**log10m_vir,
            c_2=c_2,
            overdens=overdens,
            redshift_object=redshift_object,
            redshift_source=redshift_source,
            inner_slope=inner_slope,
        )

        self.virial_radius = virial_radius
        self.overdens = overdens

        super().__init__(
            centre=centre,
            kappa_s=kappa_s,
            inner_slope=inner_slope,
            scale_radius=scale_radius,
        )
