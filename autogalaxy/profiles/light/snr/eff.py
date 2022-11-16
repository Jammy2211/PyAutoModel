from typing import Tuple

from autogalaxy.profiles.light.snr.abstract import LightProfileSNR
from autogalaxy.profiles.light import base as lp


class EllEff(lp.EllEff, LightProfileSNR):
    def __init__(
        self,
        signal_to_noise_ratio: float = 10.0,
        centre: Tuple[float, float] = (0.0, 0.0),
        elliptical_comps: Tuple[float, float] = (0.0, 0.0),
        effective_radius: float = 0.6,
        eta: float = 1.5,
    ):
        """
        The elliptical eff light profile, which is commonly used to represent the clumps of Lyman-alpha emitter
        galaxies.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        elliptical_comps
            The first and second ellipticity components of the elliptical coordinate system, (see the module
            `autogalaxy -> convert.py` for the convention).
        intensity
            Overall intensity normalisation of the light profile (units are dimensionless and derived from the data
            the light profile's image is compared too, which is expected to be electrons per second).
        effective_radius
            The circular radius containing half the light of this profile.
        eta
            Scales the intensity gradient of the profile.
        """

        super().__init__(
            centre=centre,
            elliptical_comps=elliptical_comps,
            intensity=0.0,
            effective_radius=effective_radius,
            eta=eta,
        )
        LightProfileSNR.__init__(self, signal_to_noise_ratio=signal_to_noise_ratio)


class SphEff(lp.SphEff, LightProfileSNR):
    def __init__(
        self,
        signal_to_noise_ratio: float = 10.0,
        centre: Tuple[float, float] = (0.0, 0.0),
        effective_radius: float = 0.6,
        eta: float = 1.5,
    ):
        """
        The spherical eff light profile, which is commonly used to represent the clumps of Lyman-alpha emitter
        galaxies.

        This profile is introduced in the following paper:

        https://arxiv.org/abs/1708.08854

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        intensity
            Overall intensity normalisation of the light profile (units are dimensionless and derived from the data
            the light profile's image is compared too, which is expected to be electrons per second).
        effective_radius
            The circular radius containing half the light of this profile.
        """

        super().__init__(
            centre=centre, intensity=0.0, effective_radius=effective_radius, eta=eta
        )
        LightProfileSNR.__init__(self, signal_to_noise_ratio=signal_to_noise_ratio)