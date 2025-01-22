import numpy as np
from typing import List

import autoarray as aa

from autogalaxy.galaxy.galaxy import Galaxy
from autogalaxy.galaxy.galaxies import Galaxies


class SimulatorInterferometer(aa.SimulatorInterferometer):
    def via_galaxies_from(self, galaxies: List[Galaxy], grid: aa.type.Grid2DLike):
        """
        Returns a realistic simulated image by applying effects to a plain simulated image.

        Parameters
        ----------
        image
            The image before simulating (e.g. the lens and source galaxies before optics blurring and Imaging read-out).
        pixel_scales
            The scale of each pixel in arc seconds
        exposure_time_map
            An arrays representing the effective exposure time of each pixel.
        psf: PSF
            An arrays describing the PSF the simulated image is blurred with.
        add_poisson_noise_to_data: Bool
            If `True` poisson noise_maps is simulated and added to the image, based on the total counts in each image
            pixel
        noise_seed: int
            A seed for random noise_maps generation
        """

        galaxies = Galaxies(galaxies=galaxies)

        image = galaxies.image_2d_from(grid=grid)

        return self.via_image_from(image=image)
