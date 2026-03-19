"""Train and evaluate ML models for synthetic fire ignition data."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
	confusion_matrix,
	f1_score,
	jaccard_score,
	precision_score,
	recall_score,
	roc_auc_score,
)
from sklearn.model_selection import train_test_split


class ModelTrainer:
	def __init__(self, csv_path: str, output_dir: str, seed: int = 42):
		self.csv_path = str(csv_path)
		self.output_dir = Path(output_dir)
		self.seed = int(seed)

		data = pd.read_csv(self.csv_path)
		if "Ignited" not in data.columns:
			raise ValueError("Input CSV must contain an 'Ignited' target column")

		x = data.drop(columns=["Ignited"])
		y = data["Ignited"]

		self.feature_names = list(x.columns)
		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
			x,
			y,
			test_size=0.2,
			stratify=y,
			random_state=self.seed,
		)

		self.model = None

	def train_random_forest(
		self, n_estimators: int = 200, max_depth: int | None = 15
	) -> object:
		model = RandomForestClassifier(
			n_estimators=n_estimators,
			max_depth=max_depth,
			random_state=self.seed,
			class_weight="balanced",
			n_jobs=-1,
		)
		model.fit(self.X_train, self.y_train)
		self.model = model
		return self.model

	def evaluate(self) -> dict:
		if self.model is None:
			raise RuntimeError("No trained model found. Call train_random_forest() first.")

		y_pred = self.model.predict(self.X_test)
		y_proba = self.model.predict_proba(self.X_test)[:, 1]

		cm = confusion_matrix(self.y_test, y_pred)
		precision = precision_score(self.y_test, y_pred, zero_division=0)
		recall = recall_score(self.y_test, y_pred, zero_division=0)
		f1 = f1_score(self.y_test, y_pred, zero_division=0)
		auc_roc = roc_auc_score(self.y_test, y_proba)
		jaccard = jaccard_score(self.y_test, y_pred, zero_division=0)

		print("Confusion Matrix:")
		print(cm)
		print(f"Precision: {precision:.6f}")
		print(f"Recall: {recall:.6f}")
		print(f"F1-Score: {f1:.6f}")
		print(f"AUC-ROC: {auc_roc:.6f}")
		print(f"Jaccard Index: {jaccard:.6f}")

		return {
			"precision": float(precision),
			"recall": float(recall),
			"f1": float(f1),
			"auc_roc": float(auc_roc),
			"jaccard": float(jaccard),
		}

	def save_model(self, filename: str = "fire_rf_model.joblib") -> str:
		if self.model is None:
			raise RuntimeError("No trained model found. Call train_random_forest() first.")

		self.output_dir.mkdir(parents=True, exist_ok=True)
		model_path = self.output_dir / filename
		joblib.dump(self.model, model_path)
		return str(model_path.resolve())

	def feature_importance_report(self) -> None:
		if self.model is None:
			raise RuntimeError("No trained model found. Call train_random_forest() first.")
		if not hasattr(self.model, "feature_importances_"):
			raise TypeError("Current model does not expose feature_importances_")

		importance_pairs = sorted(
			zip(self.feature_names, self.model.feature_importances_),
			key=lambda pair: pair[1],
			reverse=True,
		)

		print("Feature Importances:")
		print(f"{'Feature':<30}Importance")
		for feature_name, score in importance_pairs:
			print(f"{feature_name:<30}{score:.6f}")


if __name__ == "__main__":
	code_dir = Path(__file__).resolve().parents[1]
	default_csv = code_dir / "dataFiles" / "synthetic_fire_dataset.csv"
	default_output_dir = code_dir / "models"

	trainer = ModelTrainer(csv_path=str(default_csv), output_dir=str(default_output_dir), seed=42)
	trainer.train_random_forest()
	metrics = trainer.evaluate()
	saved_path = trainer.save_model()

	print("Saved model:", saved_path)
	print("Metrics:", metrics)
	trainer.feature_importance_report()
