"""Cellular automata fire spread engine with vectorized timestep updates."""

import joblib
import numpy as np
from scipy.ndimage import convolve


STATE_NON_BURNABLE = np.int8(1)
STATE_NOT_YET_BURNING = np.int8(2)
STATE_IGNITED = np.int8(3)
STATE_BLAZING = np.int8(4)
STATE_EXTINGUISHED = np.int8(5)


class FireAutomata:
	def __init__(self, environment: dict, config: dict):
		self.environment = environment
		self.config = config

		self.slope_risk = environment["slope_risk"]
		self.proximity_risk = environment["proximity_risk"]
		self.building_presence = environment["building_presence"]
		self.burnable_mask = environment["burnable_mask"]
		self.nodata_mask = environment["nodata_mask"]
		self.grid_shape = environment["grid_shape"]
		self.transform = environment["transform"]
		self.crs = environment["crs"]

		self.simulation_cfg = config.get("simulation", {})
		self.wind_cfg = config.get("wind", {})
		self.transition_cfg = config.get("placeholder_transition", {})
		self.ml_cfg = config.get("ml_model", {})

		self.grid = np.full(self.grid_shape, STATE_NOT_YET_BURNING, dtype=np.int8)
		self.grid[~self.burnable_mask] = STATE_NON_BURNABLE

		self.rng = np.random.default_rng(self.simulation_cfg["seed"])
		self.model = None
		self._ml_enabled = False
		self.timestep = 0

		self._precompute_base_probability()

	def _precompute_base_probability(self) -> None:
		base_ignition_prob = np.float32(self.transition_cfg.get("base_ignition_prob", 0.0))
		slope_weight = np.float32(self.transition_cfg.get("slope_weight", 0.0))
		building_weight = np.float32(self.transition_cfg.get("building_weight", 0.0))
		proximity_weight = np.float32(self.transition_cfg.get("proximity_weight", 0.0))

		self.p_base = np.clip(
			base_ignition_prob
			+ slope_weight * self.slope_risk
			+ building_weight * self.building_presence
			+ proximity_weight * self.proximity_risk,
			0.0,
			1.0,
		).astype(np.float32)

	def set_ignition(self, points: list[tuple[int, int]]) -> None:
		rows, cols = self.grid_shape
		for row, col in points:
			if row < 0 or col < 0 or row >= rows or col >= cols:
				print(f"Warning: Ignition point ({row}, {col}) is out of bounds.")
				continue
			if self.grid[row, col] == STATE_NON_BURNABLE:
				print(f"Warning: Ignition point ({row}, {col}) is non-burnable.")
				continue
			if self.grid[row, col] == STATE_NOT_YET_BURNING:
				self.grid[row, col] = STATE_BLAZING

	def step(self) -> None:
		kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.int8)
		burnout_prob = float(self.transition_cfg.get("burnout_prob", 0.0))
		wind_weight = float(self.transition_cfg.get("wind_weight", 0.0))
		wind_multiplier = 1.0 + wind_weight

		current_grid = self.grid
		next_grid = current_grid.copy()

		ignited_now = current_grid == STATE_IGNITED
		blazing_now = current_grid == STATE_BLAZING

		next_grid[ignited_now] = STATE_BLAZING

		if np.any(blazing_now):
			burnout_draw = self.rng.random(self.grid_shape)
			next_grid[blazing_now & (burnout_draw < burnout_prob)] = STATE_EXTINGUISHED

		blazing_neighbor_count = convolve(
			blazing_now.astype(np.int8),
			kernel,
			mode="constant",
			cval=0,
		)
		susceptible = (current_grid == STATE_NOT_YET_BURNING) & (blazing_neighbor_count > 0)

		if np.any(susceptible):
			if self.model is not None:
				p_ignite = self._predict_with_model()
			else:
				p_ignite = self.p_base * (blazing_neighbor_count.astype(np.float32) / np.float32(8.0))

			p_effective = np.clip(p_ignite * wind_multiplier, 0.0, 1.0)
			ignition_draw = self.rng.random(self.grid_shape)
			ignite_mask = susceptible & (ignition_draw < p_effective)
			next_grid[ignite_mask] = STATE_IGNITED

		self.grid = next_grid
		self.timestep += 1

	def load_model(self, model_path: str) -> None:
		loaded_model = joblib.load(model_path)
		predict_proba = getattr(loaded_model, "predict_proba", None)
		if predict_proba is None or not callable(predict_proba):
			raise TypeError("Loaded model must provide a callable predict_proba attribute")
		self.model = loaded_model
		self._ml_enabled = True

	def _predict_with_model(self) -> np.ndarray:
		"""Assembles per-cell feature vectors and calls self.model.predict_proba(). To be implemented when feature_pipeline.py is ready."""
		raise NotImplementedError("ML feature assembly not yet implemented")

	def is_active(self) -> bool:
		return bool(np.any((self.grid == STATE_IGNITED) | (self.grid == STATE_BLAZING)))

	def get_state_counts(self) -> dict[str, int]:
		return {
			"non_burnable": int(np.count_nonzero(self.grid == STATE_NON_BURNABLE)),
			"not_yet_burning": int(np.count_nonzero(self.grid == STATE_NOT_YET_BURNING)),
			"ignited": int(np.count_nonzero(self.grid == STATE_IGNITED)),
			"blazing": int(np.count_nonzero(self.grid == STATE_BLAZING)),
			"extinguished": int(np.count_nonzero(self.grid == STATE_EXTINGUISHED)),
		}

	def get_grid(self) -> np.ndarray:
		return self.grid.copy()
