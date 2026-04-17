import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


@dataclass
class FitResult:
    slope: float
    intercept: float
    r2: float
    n_points: int
    start_time: float
    end_time: float


def fit_window(x: np.ndarray, y: np.ndarray) -> FitResult:
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 if ss_tot == 0 else 1 - ss_res / ss_tot
    return FitResult(
        slope=float(slope),
        intercept=float(intercept),
        r2=float(r2),
        n_points=len(x),
        start_time=float(x[0]),
        end_time=float(x[-1]),
    )


def best_linear_region(group: pd.DataFrame, time_col: str, signal_col: str, min_points: int) -> FitResult:
    ordered = group.sort_values(time_col)
    x_all = ordered[time_col].to_numpy(dtype=float)
    y_all = ordered[signal_col].to_numpy(dtype=float)

    if len(ordered) < min_points:
        raise ValueError(f"Need at least {min_points} points for fitting, got {len(ordered)}")

    best: Optional[FitResult] = None
    n = len(ordered)
    for i in range(0, n - min_points + 1):
        for j in range(i + min_points, n + 1):
            fit = fit_window(x_all[i:j], y_all[i:j])
            if best is None:
                best = fit
                continue
            candidate = (fit.r2, fit.n_points, -fit.start_time)
            current = (best.r2, best.n_points, -best.start_time)
            if candidate > current:
                best = fit

    if best is None:
        raise RuntimeError("No linear region found")
    return best


def parse_well_treatment_map(spec: str) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    chunks = [c.strip() for c in spec.split(";") if c.strip()]
    for chunk in chunks:
        if ":" not in chunk:
            raise ValueError(f"Bad treatment map chunk: {chunk}")
        label, wells = chunk.split(":", 1)
        label = label.strip()
        for part in [w.strip() for w in wells.split(",") if w.strip()]:
            if "-" in part:
                start, end = part.split("-", 1)
                for w in range(int(start), int(end) + 1):
                    mapping[w] = label
            else:
                mapping[int(part)] = label
    return mapping


def parse_row_treatment_map(spec: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    chunks = [c.strip() for c in spec.split(";") if c.strip()]
    for chunk in chunks:
        if ":" not in chunk:
            raise ValueError(f"Bad row treatment map chunk: {chunk}")
        rows, label = chunk.split(":", 1)
        label = label.strip()
        for part in [r.strip().upper() for r in rows.split(",") if r.strip()]:
            if "-" in part:
                start, end = part.split("-", 1)
                for r_ord in range(ord(start), ord(end) + 1):
                    mapping[chr(r_ord)] = label
            else:
                mapping[part] = label
    return mapping


def find_time_column(df: pd.DataFrame) -> str:
    preferred = ["Row\ntime (min)", "Row time (min)", "time_min", "time"]
    for col in preferred:
        if col in df.columns:
            return col
    for col in df.columns:
        if "min" in str(col).lower() and "time" in str(col).lower():
            return col
    raise ValueError("Could not find time-in-minutes column")


def melt_plate_csv(path: Path, enzyme_label: str, time_col_name: Optional[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    time_col = time_col_name or find_time_column(df)

    plate_cols = [c for c in df.columns if ":" in str(c)]
    if not plate_cols:
        raise ValueError(f"No plate columns found in {path.name}")

    long_df = df.melt(
        id_vars=[time_col],
        value_vars=plate_cols,
        var_name="plate_well",
        value_name="absorbance",
    )
    long_df["enzyme"] = enzyme_label
    long_df["time_min"] = pd.to_numeric(long_df[time_col], errors="coerce")
    long_df["absorbance"] = pd.to_numeric(long_df["absorbance"], errors="coerce")
    long_df[["sample_row", "well_number"]] = long_df["plate_well"].astype(str).str.split(":", expand=True)
    long_df["well_number"] = long_df["well_number"].astype(int)
    long_df = long_df.dropna(subset=["time_min", "absorbance"])
    return long_df[["enzyme", "time_min", "sample_row", "well_number", "plate_well", "absorbance"]]


def blank_correct(long_df: pd.DataFrame, blank_well: int) -> pd.DataFrame:
    blank = long_df[long_df["well_number"] == blank_well][
        ["enzyme", "sample_row", "time_min", "absorbance"]
    ].rename(columns={"absorbance": "blank_absorbance"})

    merged = long_df.merge(
        blank,
        on=["enzyme", "sample_row", "time_min"],
        how="left",
        validate="many_to_one",
    )
    merged["corrected_absorbance"] = merged["absorbance"] - merged["blank_absorbance"]
    return merged


def aggregate_by_treatment(
    corrected: pd.DataFrame,
    blank_well: int,
    row_treatment_map: Optional[Dict[str, str]] = None,
    well_treatment_map: Optional[Dict[int, str]] = None,
) -> pd.DataFrame:
    work = corrected.copy()
    if row_treatment_map is not None:
        work["treatment"] = work["sample_row"].map(row_treatment_map)
    elif well_treatment_map is not None:
        work["treatment"] = work["well_number"].map(well_treatment_map)
    else:
        raise ValueError("Either row_treatment_map or well_treatment_map must be provided")

    work = work[work["well_number"] != blank_well]
    work = work.dropna(subset=["treatment"])
    agg = work[["enzyme", "sample_row", "plate_well", "well_number", "treatment", "time_min", "corrected_absorbance"]].copy()
    return agg


def reaction_rates(agg: pd.DataFrame, min_points: int, linear_start: Optional[float], linear_end: Optional[float]) -> pd.DataFrame:
    rows = []
    for (enzyme, sample, treatment, plate_well), group in agg.groupby(
        ["enzyme", "sample_row", "treatment", "plate_well"], dropna=False
    ):
        fit_df = group.copy()
        if linear_start is not None:
            fit_df = fit_df[fit_df["time_min"] >= linear_start]
        if linear_end is not None:
            fit_df = fit_df[fit_df["time_min"] <= linear_end]

        fit = best_linear_region(fit_df, time_col="time_min", signal_col="corrected_absorbance", min_points=min_points)
        rows.append(
            {
                "enzyme": enzyme,
                "sample_row": sample,
                "plate_well": plate_well,
                "treatment": treatment,
                "rate_slope": fit.slope,
                "intercept": fit.intercept,
                "r2": fit.r2,
                "n_points": fit.n_points,
                "fit_start_time_min": fit.start_time,
                "fit_end_time_min": fit.end_time,
            }
        )
    return pd.DataFrame(rows)


def summarize_rates(rates: pd.DataFrame) -> pd.DataFrame:
    return (
        rates.groupby(["enzyme", "treatment"], dropna=False)["rate_slope"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_rate", "std": "sd_rate", "count": "n"})
    )


def run_ttests(rates: pd.DataFrame, control_label: str, equal_var: bool) -> pd.DataFrame:
    rows = []
    for enzyme, e_df in rates.groupby("enzyme", dropna=False):
        control = e_df[e_df["treatment"] == control_label]["rate_slope"].dropna()
        if len(control) < 2:
            continue

        for treatment, t_df in e_df.groupby("treatment", dropna=False):
            if treatment == control_label:
                continue
            vals = t_df["rate_slope"].dropna()
            if len(vals) < 2:
                continue

            t_stat, p_value = stats.ttest_ind(vals, control, equal_var=equal_var, nan_policy="omit")
            rows.append(
                {
                    "enzyme": enzyme,
                    "comparison": f"{treatment} vs {control_label}",
                    "n_treatment": len(vals),
                    "n_control": len(control),
                    "t_stat": t_stat,
                    "p_value": p_value,
                    "significant_alpha_0_05": p_value < 0.05,
                    "test_type": "Student two-sample t-test" if equal_var else "Welch two-sample t-test",
                }
            )
    return pd.DataFrame(rows)


def plot_timecourses(agg: pd.DataFrame, rates: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for (enzyme, treatment), group in agg.groupby(["enzyme", "treatment"], dropna=False):
        fig, ax = plt.subplots(figsize=(8.5, 5))
        for plate_well, g in group.groupby("plate_well", dropna=False):
            g = g.sort_values("time_min")
            sample_line = ax.plot(
                g["time_min"],
                g["corrected_absorbance"],
                linewidth=1.0,
                alpha=0.5,
                label=f"{plate_well}",
            )[0]
            row = rates[
                (rates["enzyme"] == enzyme)
                & (rates["treatment"] == treatment)
                & (rates["plate_well"] == plate_well)
            ]
            if not row.empty:
                slope = float(row["rate_slope"].iloc[0])
                intercept = float(row["intercept"].iloc[0])
                start_t = float(row["fit_start_time_min"].iloc[0])
                end_t = float(row["fit_end_time_min"].iloc[0])
                x = np.array([start_t, end_t])
                y = slope * x + intercept
                ax.plot(
                    x,
                    y,
                    linestyle="--",
                    linewidth=1.8,
                    color=sample_line.get_color(),
                    alpha=0.95,
                )

        ax.set_title(f"Corrected Absorbance vs Time: {enzyme} | {treatment}")
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Corrected absorbance (AU)")
        ax.legend(fontsize=6, ncol=4)
        fig.tight_layout()
        safe = f"timecourse_{enzyme}_{treatment}".replace(" ", "_")
        fig.savefig(out_dir / f"{safe}.png", dpi=220)
        plt.close(fig)


def plot_summary_bar(summary: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=summary, x="treatment", y="mean_rate", hue="enzyme", errorbar=None, ax=ax)

    for i, row in summary.iterrows():
        patch = ax.patches[i]
        x = patch.get_x() + patch.get_width() / 2
        yerr = row["sd_rate"] if pd.notna(row["sd_rate"]) else 0.0
        ax.errorbar(x=x, y=row["mean_rate"], yerr=yerr, fmt="none", ecolor="black", capsize=4, linewidth=1.2)

    ax.set_title("Reaction Rate Means ± SD by Treatment and Enzyme")
    ax.set_xlabel("Treatment")
    ax.set_ylabel("Reaction rate (ΔAbs/min)")
    fig.tight_layout()
    fig.savefig(out_dir / "rate_means_sd_barplot.png", dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze AChE/BChE assay plate CSV files")
    parser.add_argument("--ache", default="AChe.csv", help="Path to AChE CSV")
    parser.add_argument("--bche", default="BChe.csv", help="Path to BChE CSV")
    parser.add_argument("--outdir", default="analysis_output", help="Output folder")
    parser.add_argument("--time-col", default=None, help="Time column name if auto-detection fails")
    parser.add_argument("--blank-well", type=int, default=1, help="Blank well number")
    parser.add_argument("--min-points", type=int, default=5, help="Min points in linear fit")
    parser.add_argument("--linear-start", type=float, default=None, help="Optional fit start time (min)")
    parser.add_argument("--linear-end", type=float, default=None, help="Optional fit end time (min)")
    parser.add_argument("--control", default="methanol", help="Control treatment for t-test")
    parser.add_argument("--assume-equal-variance", action="store_true", help="Use Student t-test instead of Welch")
    parser.add_argument(
        "--treatment-map",
        default="plant_essential_oils:2-5;malaoxon:6-8;methanol:9-12",
        help="Map well numbers to treatments, e.g. 'plant_essential_oils:2-5;malaoxon:6-8;methanol:9-12'",
    )
    parser.add_argument(
        "--row-treatment-map",
        default="A-F:plant_essential_oils;G:malaoxon;H:methanol",
        help="Map plate rows to treatments, e.g. 'A-F:plant_essential_oils;G:malaoxon;H:methanol'",
    )
    parser.add_argument(
        "--map-mode",
        choices=["row", "well"],
        default="row",
        help="Treatment mapping mode: row-based (default) or well-number-based",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sns.set_theme(style="whitegrid")

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    row_treatment_map = parse_row_treatment_map(args.row_treatment_map)
    well_treatment_map = parse_well_treatment_map(args.treatment_map)

    ache = melt_plate_csv(Path(args.ache), enzyme_label="AChE", time_col_name=args.time_col)
    bche = melt_plate_csv(Path(args.bche), enzyme_label="BChE", time_col_name=args.time_col)
    combined = pd.concat([ache, bche], ignore_index=True)

    corrected = blank_correct(combined, blank_well=args.blank_well)
    corrected.to_csv(out_dir / "corrected_absorbance_by_well.csv", index=False)

    agg = aggregate_by_treatment(
        corrected,
        blank_well=args.blank_well,
        row_treatment_map=row_treatment_map if args.map_mode == "row" else None,
        well_treatment_map=well_treatment_map if args.map_mode == "well" else None,
    )
    agg.to_csv(out_dir / "corrected_absorbance_by_sample_treatment.csv", index=False)

    rates = reaction_rates(
        agg,
        min_points=args.min_points,
        linear_start=args.linear_start,
        linear_end=args.linear_end,
    )
    rates.to_csv(out_dir / "reaction_rates_by_sample.csv", index=False)

    summary = summarize_rates(rates)
    summary.to_csv(out_dir / "reaction_rate_summary_mean_sd.csv", index=False)

    ttests = run_ttests(rates, control_label=args.control, equal_var=args.assume_equal_variance)
    ttests.to_csv(out_dir / "ttest_results.csv", index=False)

    plot_timecourses(agg, rates, out_dir=out_dir)
    plot_summary_bar(summary, out_dir=out_dir)

    with open(out_dir / "run_notes.txt", "w", encoding="utf-8") as f:
        f.write(f"Mapping mode: {args.map_mode}\n")
        if args.map_mode == "row":
            f.write("Assumed row treatment map:\n")
            for r, t in sorted(row_treatment_map.items()):
                f.write(f"  Row {r}: {t}\n")
        else:
            f.write("Assumed well treatment map:\n")
            for w, t in sorted(well_treatment_map.items()):
                f.write(f"  Well {w}: {t}\n")
        f.write("\nReaction rate unit: absorbance units/min\n")
        f.write("t-tests compare each treatment to control label provided.\n")

    print("Analysis complete")
    print(f"Output folder: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
