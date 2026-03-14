"""Assemble per-cell environmental and dynamic features for ML fire-ignition prediction."""

from __future__ import annotations

import numpy as np


class FeatureAssembler:
	FEATURE_NAMES = [
		"slope_risk",
		"proximity_risk",
		"building_presence",
		"wind_speed",
		"wind_sin",
		"wind_cos",
		"neighbor_burning_count",
		"composite_flammability",
	]

	def __init__(self, environment: dict, wind_config: dict):
		self.slope_risk = environment["slope_risk"]
		self.proximity_risk = environment["proximity_risk"]
		self.building_presence = environment["building_presence"]
		self.burnable_mask = environment["burnable_mask"]
		self.grid_shape = environment["grid_shape"]

		direction_deg = float(wind_config["direction_deg"])
		radians = np.deg2rad(direction_deg)

		self.wind_dx = float(np.sin(radians))
		self.wind_dy = float(-np.cos(radians))

		self.wind_speed = float(wind_config["speed_kmh"])
		self.wind_sin = float(np.sin(radians))
		self.wind_cos = float(np.cos(radians))

		self.feature_names = list(self.FEATURE_NAMES)

	def assemble_grid_features(self, blazing_neighbor_count: np.ndarray) -> np.ndarray:
		if blazing_neighbor_count.shape != self.grid_shape:
			raise ValueError(
				"blazing_neighbor_count shape mismatch: "
				f"{blazing_neighbor_count.shape} != {self.grid_shape}"
			)

		composite_flammability = self.building_presence * (
			np.float32(0.5)
			+ np.float32(0.3) * self.slope_risk
			+ np.float32(0.2) * self.proximity_risk
		)

		n_cells = int(np.prod(self.grid_shape))
		wind_speed_col = np.full(n_cells, self.wind_speed, dtype=np.float32)
		wind_sin_col = np.full(n_cells, self.wind_sin, dtype=np.float32)
		wind_cos_col = np.full(n_cells, self.wind_cos, dtype=np.float32)

		features = np.column_stack(
			[
				self.slope_risk.ravel(),
				self.proximity_risk.ravel(),
				self.building_presence.ravel(),
				wind_speed_col,
				wind_sin_col,
				wind_cos_col,
				blazing_neighbor_count.ravel(),
				composite_flammability.ravel(),
			]
		).astype(np.float32)

		return features

	def assemble_masked_features(
		self, blazing_neighbor_count: np.ndarray
	) -> tuple[np.ndarray, np.ndarray]:
		features_full = self.assemble_grid_features(blazing_neighbor_count)
		burnable_flat = self.burnable_mask.ravel().astype(bool)
		mask_indices = np.flatnonzero(burnable_flat)
		features_masked = features_full[burnable_flat]
		return features_masked, mask_indices
