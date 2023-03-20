from typing import Optional

import autoarray.plot as aplt


class Include2D(aplt.Include2D):
    def __init__(
        self,
        origin=None,
        mask=None,
        border=None,
        grid=None,
        positions=None,
        light_profile_centres=None,
        mass_profile_centres=None,
        tangential_critical_curves=None,
        radial_critical_curves=None,
        tangential_caustics=None,
        radial_caustics=None,
        multiple_images=None,
        mapper_source_plane_mesh_grid: Optional[bool] = None,
        mapper_source_plane_data_grid: Optional[bool] = None,
        mapper_image_plane_mesh_grid=None,
    ):

        super().__init__(
            origin=origin,
            mask=mask,
            border=border,
            grid=grid,
            mapper_source_plane_mesh_grid=mapper_source_plane_mesh_grid,
            mapper_source_plane_data_grid=mapper_source_plane_data_grid,
            mapper_image_plane_mesh_grid=mapper_image_plane_mesh_grid,
        )

        self._positions = positions
        self._light_profile_centres = light_profile_centres
        self._mass_profile_centres = mass_profile_centres
        self._tangential_critical_curves = tangential_critical_curves
        self._radial_critical_curves = radial_critical_curves
        self._tangential_caustics = tangential_caustics
        self._radial_caustics = radial_caustics
        self._multiple_images = multiple_images

    @property
    def positions(self):
        return self.load(value=self._positions, name="positions")

    @property
    def light_profile_centres(self):
        return self.load(
            value=self._light_profile_centres, name="light_profile_centres"
        )

    @property
    def mass_profile_centres(self):
        return self.load(value=self._mass_profile_centres, name="mass_profile_centres")

    @property
    def tangential_critical_curves(self):
        return self.load(
            value=self._tangential_critical_curves, name="tangential_critical_curves"
        )

    @property
    def radial_critical_curves(self):
        return self.load(
            value=self._radial_critical_curves, name="radial_critical_curves"
        )

    @property
    def tangential_caustics(self):
        return self.load(value=self._tangential_caustics, name="tangential_caustics")

    @property
    def radial_caustics(self):
        return self.load(value=self._radial_caustics, name="radial_caustics")

    @property
    def multiple_images(self):
        return self.load(value=self._multiple_images, name="multiple_images")
