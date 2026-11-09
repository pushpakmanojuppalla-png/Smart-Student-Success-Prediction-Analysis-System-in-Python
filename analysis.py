"""Exploratory data analysis, feature importance and an auto-written insights report."""
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402

from src import config  # noqa: E402
from src.preprocessing import split_data  # noqa: E402

COLORS = {"At-Risk": "#d1495b", "Average": "#edae49", "High": "#2e8b57"}


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def run_eda(df: pd.DataFrame, reports_dir=config.REPORTS_DIR) -> None:
    reports_dir = Path(reports_dir)
    fig_dir = reports_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Score distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df[config.TARGET_SCORE], bins=30, color="#3d5a80", edgecolor="white")
    ax.axvline(config.AT_RISK_THRESHOLD, color="#d1495b", ls="--", label="At-Risk cutoff")
    ax.axvline(config.HIGH_THRESHOLD, color="#2e8b57", ls="--", label="High cutoff")
    ax.set(title="Final score distribution", xlabel="Final score", ylabel="Students")
    ax.legend()
    _save(fig, fig_dir / "score_distribution.png")

    # 2. Performance level counts
    counts = df[config.TARGET_LEVEL].value_counts().reindex(config.LEVELS)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(counts.index, counts.values, color=[COLORS[l] for l in counts.index])
    for i, v in enumerate(counts.values):
        ax.text(i, v, str(v), ha="center", va="bottom")
    ax.set(title="Students by performance level", ylabel="Students")
    _save(fig, fig_dir / "level_counts.png")

    # 3. Correlation heatmap
    cols = config.NUMERIC_FEATURES + [config.TARGET_SCORE]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=60, ha="right")
    ax.set_yticks(range(len(cols)), cols)
    ax.set_title("Correlation matrix")
    fig.colorbar(im, ax=ax, fraction=0.046)
    _save(fig, fig_dir / "correlation_heatmap.png")

    # 4. Key relationships with the score
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col in zip(axes, ["attendance_pct", "study_hours_per_week", "previous_gpa"]):
        for level in config.LEVELS:
            sub = df[df[config.TARGET_LEVEL] == level]
            ax.scatter(sub[col], sub[config.TARGET_SCORE], s=8, alpha=0.5,
                       color=COLORS[level], label=level)
        ax.set(xlabel=col, ylabel="final_score", title=f"{col} vs score")
    axes[0].legend()
    _save(fig, fig_dir / "key_relationships.png")

    # 5. Score by family support
    order = ["Low", "Medium", "High"]
    data = [df.loc[df["family_support"] == g, config.TARGET_SCORE].dropna() for g in order]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.boxplot(data, tick_labels=order) if _supports_tick_labels() else ax.boxplot(data, labels=order)
    ax.set(title="Final score by family support", ylabel="Final score")
    _save(fig, fig_dir / "score_by_family_support.png")

    # 6. At-risk rate by attendance band
    bands = pd.cut(df["attendance_pct"], [0, 60, 70, 80, 90, 100])
    rate = (df[config.TARGET_LEVEL] == "At-Risk").groupby(bands, observed=True).mean() * 100
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([str(i) for i in rate.index], rate.values, color="#d1495b")
    ax.set(title="At-Risk rate by attendance band", xlabel="Attendance %",
           ylabel="% of students At-Risk")
    _save(fig, fig_dir / "at_risk_by_attendance.png")

    # Tables
    df[config.NUMERIC_FEATURES + [config.TARGET_SCORE]].describe().round(2).to_csv(
        reports_dir / "summary_stats.csv")
    df.groupby(config.TARGET_LEVEL)[config.NUMERIC_FEATURES + [config.TARGET_SCORE]] \
        .mean().round(2).reindex(config.LEVELS).to_csv(reports_dir / "level_profile.csv")


def _supports_tick_labels() -> bool:
    # matplotlib >= 3.9 renamed boxplot's `labels` to `tick_labels`
    major, minor = (int(x) for x in matplotlib.__version__.split(".")[:2])
    return (major, minor) >= (3, 9)


def run_feature_importance(df: pd.DataFrame, models_dir=config.MODELS_DIR,
                           reports_dir=config.REPORTS_DIR) -> pd.DataFrame:
    """Permutation importance of the saved classifier on the held-out test split."""
    reports_dir = Path(reports_dir)
    fig_dir = reports_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    clf = joblib.load(Path(models_dir) / "classifier.joblib")
    _, test_df = split_data(df)
    result = permutation_importance(
        clf, test_df[config.FEATURES], test_df[config.TARGET_LEVEL],
        scoring="f1_macro", n_repeats=10, random_state=config.RANDOM_STATE)
    imp = (pd.DataFrame({"feature": config.FEATURES,
                         "importance": result.importances_mean,
                         "std": result.importances_std})
           .sort_values("importance", ascending=False).reset_index(drop=True))
    imp.to_csv(reports_dir / "feature_importance.csv", index=False)

    top = imp.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top["feature"], top["importance"], xerr=top["std"], color="#3d5a80")
    ax.set(title="What drives the prediction? (permutation importance)",
           xlabel="Drop in macro-F1 when feature is shuffled")
    _save(fig, fig_dir / "feature_importance.png")
    return imp


def write_insights(df: pd.DataFrame, importance: pd.DataFrame | None = None,
                   reports_dir=config.REPORTS_DIR) -> str:
    reports_dir = Path(reports_dir)
    n = len(df)
    level_share = (df[config.TARGET_LEVEL].value_counts(normalize=True) * 100) \
        .reindex(config.LEVELS).round(1)
    corr = df[config.NUMERIC_FEATURES + [config.TARGET_SCORE]].corr()[config.TARGET_SCORE] \
        .drop(config.TARGET_SCORE).sort_values(key=np.abs, ascending=False)

    lines = ["# Student Success - Automated Insights", "",
             f"- Students analysed: **{n}**",
             f"- Mean final score: **{df[config.TARGET_SCORE].mean():.1f}** "
             f"(median {df[config.TARGET_SCORE].median():.1f})", ""]
    lines.append("## Performance levels")
    for lvl in config.LEVELS:
        lines.append(f"- {lvl}: {level_share[lvl]}%")
    lines += ["", "## Strongest linear correlations with final score"]
    for feat, val in corr.head(5).items():
        lines.append(f"- `{feat}`: {val:+.2f}")

    low_att = df[df["attendance_pct"] < 70]
    if len(low_att):
        rate = (low_att[config.TARGET_LEVEL] == "At-Risk").mean() * 100
        lines += ["", "## Early-warning signals",
                  f"- Students with attendance below 70%: **{rate:.0f}%** are At-Risk "
                  f"(overall: {level_share['At-Risk']}%)."]
    if importance is not None:
        lines += ["", "## Most influential features (model)"]
        for _, r in importance.head(5).iterrows():
            lines.append(f"- `{r['feature']}` (importance {r['importance']:.3f})")

    text = "\n".join(lines) + "\n"
    (reports_dir / "insights.md").write_text(text)
    return text
