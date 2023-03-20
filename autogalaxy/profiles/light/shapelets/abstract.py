from typing import Tuple

from autogalaxy.profiles.light.linear.abstract import LightProfileLinear


class AbstractShapelet(LightProfileLinear):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        beta: float = 1.0,
    ):
        """
        Abstract Base class of a Shapelet.

        Shapelets are always defined and used as a linear light profile.

        Shapelets are defined according to:

          https://arxiv.org/abs/astro-ph/0105178

        Shapelets are are described in the context of strong lens modeling in:

          https://ui.adsabs.harvard.edu/abs/2016MNRAS.457.3066T/abstract

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile (shapelet) centre.
        ell_comps
            The first and second ellipticity components of the elliptical coordinate system.
        beta
            The characteristic length scale of the shapelet basis function, defined in arc-seconds.
        """

        self.beta = beta

        super().__init__(centre=centre, ell_comps=ell_comps, intensity=1.0)
