import inspect
import numpy as np
from typing import Dict, List, Optional, Tuple

from autoconf import cached_property
import autoarray as aa
import autofit as af

from autogalaxy.profiles.light_profiles.light_profiles_operated import (
    LightProfileOperated,
)

from autogalaxy.profiles.light_profiles import light_profiles as lp
from autogalaxy.profiles import light_and_mass_profiles as lmp

from autogalaxy import exc


class EllMoffat(lp.EllMoffat, LightProfileLinear):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        elliptical_comps: Tuple[float, float] = (0.0, 0.0),
        alpha: float = 0.5,
        beta: float = 2.0,
    ):
        """
        The elliptical Moffat light profile, which is commonly used to model the Point Spread Function of
        Astronomy observations.

        This form of the MOffat profile is a reparameterizaiton of the original formalism given by
        https://ui.adsabs.harvard.edu/abs/1969A%26A.....3..455M/abstract. The actual profile itself is identical.

        See `autogalaxy.profiles.light_profiles.light_profiles.LightProfile` for a description of light profile objects.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        elliptical_comps
            The first and second ellipticity components of the elliptical coordinate system, (see the module
        alpha
            Scales the overall size of the Moffat profile and for a PSF typically corresponds to the FWHM / 2.
        beta
            Scales the wings at the outskirts of the Moffat profile, where smaller values imply heavier wings and it
            tends to a Gaussian as beta goes to infinity.
        """

        super().__init__(
            centre=centre,
            elliptical_comps=elliptical_comps,
            intensity=1.0,
            alpha=alpha,
            beta=beta,
        )

    @property
    def lp_cls(self):
        return lp.EllMoffat

    @property
    def lmp_cls(self):
        return lmp.EllGaussian
