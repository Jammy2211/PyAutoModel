import numpy as np
from typing import Optional, Tuple

import autoarray as aa

from autogalaxy.profiles.snr.abstract import LightProfileSNR
from autogalaxy.profiles.light_profiles import light_profiles as lp
from scipy.optimize import root_scalar


class EllSersic(lp.EllSersic, LightProfileSNR):
    def __init__(
        self,
        signal_to_noise_ratio: float = 10.0,
        centre: Tuple[float, float] = (0.0, 0.0),
        elliptical_comps: Tuple[float, float] = (0.0, 0.0),
        effective_radius: float = 0.6,
        sersic_index: float = 4.0,
    ):
        """
        An elliptical Sersic light profile.

        Instead of an `intensity` a `signal_to_noise_ratio` is input which sets the signal to noise of the brightest
        pixel of the profile's image when used to simulate imaging data.

        Parameters
        ----------
        signal_to_noise_ratio
            The signal to noise of the light profile when it is used to simulate strong lens imaging.
        centre
            The (y,x) arc-second coordinates of the profile centre.
        elliptical_comps
            The first and second ellipticity components of the elliptical coordinate system, (see the module
            `autogalaxy -> convert.py` for the convention).
        effective_radius
            The circular radius containing half the light of this profile.
        sersic_index
            Controls the concentration of the profile (lower -> less concentrated, higher -> more concentrated).
        """
        super().__init__(
            centre=centre,
            elliptical_comps=elliptical_comps,
            intensity=0.0,
            effective_radius=effective_radius,
            sersic_index=sersic_index,
        )

        LightProfileSNR.__init__(self, signal_to_noise_ratio=signal_to_noise_ratio)


class SphSersic(lp.SphSersic, LightProfileSNR):
    def __init__(
        self,
        signal_to_noise_ratio: float = 10.0,
        centre: Tuple[float, float] = (0.0, 0.0),
        effective_radius: float = 0.6,
        sersic_index: float = 4.0,
    ):
        """
        The spherical Sersic light profile.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        intensity
            Overall intensity normalisation of the light profile (units are dimensionless and derived from the data
            the light profile's image is compared too, which is expected to be electrons per second).
        effective_radius
            The circular radius containing half the light of this profile.
        sersic_index
            Controls the concentration of the of the light profile.
        """
        super().__init__(
            centre=centre,
            intensity=0.0,
            effective_radius=effective_radius,
            sersic_index=sersic_index,
        )
        LightProfileSNR.__init__(self, signal_to_noise_ratio=signal_to_noise_ratio)