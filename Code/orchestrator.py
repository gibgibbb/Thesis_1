from pathlib import Path

import numpy as np
import rasterio
import yaml

from modules.automata_engine import FireAutomata
from modules.data_loader import EnvironmentManager


def load_config(config_path: str) -> dict:
	config_file = Path(config_path)
	with config_file.open("r", encoding="utf-8") as file_obj:
		config = yaml.safe_load(file_obj)

	code_dir = Path(__file__).resolve().parent
	raster_dir = config["environment"]["raster_dir"]
	config["environment"]["raster_dir"] = str((code_dir / raster_dir).resolve())
	return config


def _pick_random_ignition_points(automata: FireAutomata, n_points: int = 3) -> list[tuple[int, int]]:
	candidate_mask = automata.burnable_mask & (automata.building_presence == 1)
	candidate_cells = np.argwhere(candidate_mask)
	if candidate_cells.size == 0:
		return []

	count = min(n_points, candidate_cells.shape[0])
	selected_idx = automata.rng.choice(candidate_cells.shape[0], size=count, replace=False)
	selected_cells = candidate_cells[selected_idx]
	return [(int(row), int(col)) for row, col in selected_cells]


def run_simulation(config: dict) -> None:
	env_manager = EnvironmentManager(config["environment"])
	env_manager.load_rasters()
	env_manager.build_masks()
	env_manager.normalize_layers()
	env_manager.summary()

	automata = FireAutomata(env_manager.get_environment(), config)

	ml_cfg = config.get("ml_model", {})
	if ml_cfg.get("enabled") is True and str(ml_cfg.get("model_path", "")).strip() != "":
		automata.load_model(ml_cfg["model_path"])

	ignition_points = config["simulation"].get("ignition_points", [])
	if ignition_points:
		ignition_tuples = [(int(row), int(col)) for row, col in ignition_points]
	else:
		ignition_tuples = _pick_random_ignition_points(automata, n_points=3)

	if not ignition_tuples:
		raise ValueError("No valid ignition points available from config or random building candidates.")

	automata.set_ignition(ignition_tuples)
	print(f"Ignition points: {ignition_tuples}")

	max_timesteps = int(config["simulation"].get("max_timesteps", 0))
	for step in range(1, max_timesteps + 1):
		automata.step()

		if step % 10 == 0:
			counts = automata.get_state_counts()
			print(f"Step {step}: {counts}")

		if not automata.is_active():
			print(f"Simulation stopped early at step {step}: no active fire cells remain.")
			break

	output_cfg = config.get("output", {})
	output_dir = Path(output_cfg.get("output_dir", "output"))
	if not output_dir.is_absolute():
		output_dir = Path(__file__).resolve().parent / output_dir
	output_dir.mkdir(parents=True, exist_ok=True)

	final_grid = automata.get_grid().astype(np.int8)
	output_path = output_dir / "final_state.tif"
	with rasterio.open(
		output_path,
		"w",
		driver="GTiff",
		height=final_grid.shape[0],
		width=final_grid.shape[1],
		count=1,
		dtype="int8",
		crs=automata.crs,
		transform=automata.transform,
	) as dst:
		dst.write(final_grid, 1)

	print(f"Final state written to: {output_path}")


if __name__ == "__main__":
	default_config_path = Path(__file__).resolve().parent / "config" / "default_experiment.yaml"
	run_simulation(load_config(str(default_config_path)))
