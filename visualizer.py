"""Visualize evaluation_results.csv as bar charts."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_results(csv_path: Path) -> pd.DataFrame:
	if not csv_path.exists():
		raise FileNotFoundError(f"Results CSV not found at {csv_path}")
	df = pd.read_csv(csv_path)
	required_cols = {
		"overall_report_score",
		"classification_score",
		"impression_score",
	}
	missing = required_cols - set(df.columns)
	if missing:
		raise ValueError(f"CSV missing required columns: {sorted(missing)}")
	return df


def summarize(df: pd.DataFrame) -> dict:
	total = len(df)
	return {
		"total": total,
		"correct_with_minor_mistakes": (df["overall_report_score"] >= 0.8).sum(),
		"correct": (df["overall_report_score"] > 0.9).sum(),
		"perfect": (df["overall_report_score"] == 1.0).sum(),
		"mean_overall": df["overall_report_score"].mean(),
	}


def plot_summary(stats: dict, output_path: Path) -> None:
	fig, ax = plt.subplots(figsize=(6, 4))

	labels = ["Acceptable\nscore >= 0.8", "Correct w/ no errors\nscore > 0.9"]
	counts = [
		stats["correct_with_minor_mistakes"],
		stats["correct"],
	]
	ax.bar(labels, counts, color=["#4c78a8", "#f58518"])
	ax.set_ylabel("Count")
	ax.set_title("Overall Report Score Counts")
	ax.set_ylim(0, max(counts + [1]))
	ax.tick_params(axis="x", rotation=20)

	fig.tight_layout()
	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(output_path, dpi=150)
	print(f"Saved visualization to {output_path}")


def main():
	csv_path = Path("logs/evaluation_results.csv")
	output_path = Path("logs/evaluation_results.png")
	df = load_results(csv_path)
	stats = summarize(df)
	if stats["total"] == 0:
		raise ValueError("No rows to visualize in results CSV")
	plot_summary(stats, output_path)


if __name__ == "__main__":
	main()
