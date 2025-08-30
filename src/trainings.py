from __future__ import annotations
import re
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import unicodedata
from typing import Optional, List, Dict, Tuple
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ----------------- Helpers: Parsing & Normalisierung -----------------

def _to_float(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    x = str(x).strip().replace(",", ".")
    try:
        return float(x)
    except:
        return np.nan

def _normalize_ascii(s: str) -> str:
    """Deutsch/UTF-8 tolerant -> ascii, Kleinbuchstaben, ohne Sonderzeichen."""
    if s is None:
        return ""
    no_diac = (
        unicodedata.normalize("NFKD", str(s))
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    clean = "".join(ch if ch.isalnum() else " " for ch in no_diac)
    clean = re.sub(r"\s+", "", clean)
    return clean.lower()

def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {}
    for col in df.columns:
        norm = _normalize_ascii(col)
        if norm == "datum":
            std = "datum"
        elif norm in ("ubung", "uebung", "übung"):
            std = "uebung"
        else:
            std = norm
        mapping[col] = std
    return df.rename(columns=mapping)

def _detect_set_columns(columns: List[str]) -> List[Tuple[str, str, int]]:
    weight_patterns = [r"^satz\D*([0-9]+)$", r"^gewicht\D*([0-9]+)$", r"^kg\D*([0-9]+)$", r"^set\D*([0-9]+)$"]
    reps_patterns   = [r"^wdh\D*([0-9]+)$", r"^reps?\D*([0-9]+)$", r"^wiederholungen\D*([0-9]+)$"]

    def _collect(cands: List[str], pats: List[str]) -> List[Tuple[str, int]]:
        hits = []
        for c in cands:
            for p in pats:
                m = re.match(p, c)
                if m:
                    try:
                        idx = int(m.group(1))
                        hits.append((c, idx))
                        break
                    except:
                        pass
        return hits

    base = {"datum", "uebung"}
    others = [c for c in columns if c not in base]
    weights = _collect(others, weight_patterns)
    reps    = _collect(others, reps_patterns)

    w_by_idx: Dict[int, str] = {}
    r_by_idx: Dict[int, str] = {}
    for name, idx in weights:
        if idx not in w_by_idx or name.startswith("satz"):
            w_by_idx[idx] = name
    for name, idx in reps:
        if idx not in r_by_idx or name.startswith("wdh"):
            r_by_idx[idx] = name

    pairs: List[Tuple[str, str, int]] = []
    for idx in sorted(set(w_by_idx).intersection(r_by_idx)):
        pairs.append((w_by_idx[idx], r_by_idx[idx], idx))
    return pairs

# ----------------- Diagnose (optional) -----------------

def detect_columns_info(df: pd.DataFrame) -> dict:
    cols = list(df.columns)
    pairs = _detect_set_columns(cols)
    return {
        "columns": cols,
        "has_datum": "datum" in cols,
        "has_uebung": "uebung" in cols,
        "pairs": [{"weight_col": w, "reps_col": r, "index": i} for (w, r, i) in pairs],
    }

# ----------------- Public API: Laden/Transformieren -----------------

def load_training_csv(path: str) -> pd.DataFrame:
    encodings = ["utf-8", "cp1252", "latin1"]
    last_err = None
    df = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc, sep=None, engine="python")
            break
        except Exception as e:
            last_err = e
            df = None
    if df is None:
        raise last_err

    df = _standardize_columns(df)

    if "datum" not in df.columns:
        raise ValueError("Spalte 'Datum' (oder Variante) nicht gefunden.")
    df["datum"] = pd.to_datetime(df["datum"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["datum"])

    if "uebung" not in df.columns:
        raise ValueError("Spalte 'Übung'/'Uebung' (oder Variante) nicht gefunden.")

    ordered = ["datum", "uebung"] + [c for c in df.columns if c not in ("datum", "uebung")]
    return df[ordered]

def to_long_form(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    pairs = _detect_set_columns(list(df.columns))
    rows = []
    for _, row in df.iterrows():
        for weight_col, reps_col, idx in pairs:
            w = _to_float(row.get(weight_col))
            r = _to_float(row.get(reps_col))
            if np.isnan(w) or np.isnan(r):
                continue
            rows.append({
                "date": row["datum"].date(),
                "exercise": str(row["uebung"]).strip(),
                "set_idx": idx,
                "weight": w,
                "reps": r,
                "volume": w * r,
                "e1rm_epley": w * (1 + r/30.0),
            })
    long = pd.DataFrame(rows)
    if long.empty:
        return long
    long["date"] = pd.to_datetime(long["date"])
    # Woche = Montag
    long["week"] = long["date"] - long["date"].dt.weekday * pd.Timedelta(days=1)
    return long

def weekly_summary(long: pd.DataFrame) -> pd.DataFrame:
    if long.empty:
        return long
    agg = (long
           .groupby(["week", "exercise"])
           .agg(volume_sum=("volume","sum"),
                hard_sets=("set_idx","count"),
                best_e1rm=("e1rm_epley","max"))
           .reset_index())
    return agg

# ----------------- Analysen -----------------

def detect_plateau(weekly: pd.DataFrame, exercise: str, weeks:int=6) -> dict:
    data = weekly[weekly["exercise"]==exercise].sort_values("week").tail(weeks)
    if len(data) < max(3, weeks//2):
        return {"enough_data": False}
    X = np.arange(len(data)).reshape(-1,1)
    y = data["best_e1rm"].values
    model = LinearRegression().fit(X,y)
    slope = float(model.coef_[0])
    return {
        "enough_data": True,
        "slope_per_week": slope,
        "is_plateau": abs(slope) < 0.25
    }

def acwr(long: pd.DataFrame, day_window_acute:int=7, day_window_chronic:int=28) -> Optional[float]:
    if long.empty:
        return None
    df = long[["date","volume"]].copy()
    df = df.groupby("date").sum().asfreq("D").fillna(0.0)
    acute = df.tail(day_window_acute)["volume"].sum()
    chronic = df.tail(day_window_chronic)["volume"].mean() if len(df)>=day_window_chronic else None
    if not chronic or chronic == 0:
        return None
    return float(acute / chronic)

def summarize_for_prompt(weekly: pd.DataFrame, top_n:int=8) -> str:
    if weekly.empty:
        return "Keine Trainingsdaten geladen."
    parts = []
    for ex, grp in weekly.groupby("exercise"):
        g = grp.sort_values("week").tail(top_n)
        if g.empty:
            continue
        vols = g['volume_sum'].tolist()
        e1s  = g['best_e1rm'].tolist()
        delta_vol = round(vols[-1] - vols[0], 1) if len(vols) > 1 else 0.0
        delta_e1  = round(e1s[-1]  - e1s[0], 1)  if len(e1s)  > 1 else 0.0
        line = (
            f"- {ex}: Volumen/Woche {[round(v,1) for v in vols]} (Δ={delta_vol}); "
            f"beste e1RM {[round(v,1) for v in e1s]} (Δ={delta_e1})"
        )
        parts.append(line)
    if not parts:
        return "Keine verwertbaren Sätze gefunden."
    return "\n".join(parts)

def per_exercise_summary(long: pd.DataFrame, weekly: pd.DataFrame) -> pd.DataFrame:
    """
    Kompakte Übersicht je Übung (sprechende Spalten + Einheiten).
    """
    if long.empty or weekly.empty:
        return pd.DataFrame()
    rows = []
    for ex, grp in weekly.groupby("exercise"):
        g = grp.sort_values("week")
        last8 = g.tail(8)
        slope = None
        plateau = None
        if len(last8) >= 3:
            X = np.arange(len(last8)).reshape(-1,1)
            y = last8["best_e1rm"].values
            slope = float(LinearRegression().fit(X,y).coef_[0])
            plateau = abs(slope) < 0.25
        rows.append({
            "Übung": ex,
            "Sätze gesamt": int(long[long["exercise"]==ex].shape[0]),
            "Ø Volumen (letzte 4 W) [kg·Wdh]": float(g.tail(4)["volume_sum"].mean()) if len(g)>=1 else 0.0,
            "Bestes e1RM (gesamter Zeitraum) [kg]": float(g["best_e1rm"].max() if len(g) else 0.0),
            "Trend e1RM (letzte 8 W) [kg/Woche]": None if slope is None else float(slope),
            "Plateau (6 W)": plateau if plateau is not None else None,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["Ø Volumen (letzte 4 W) [kg·Wdh]"] = out["Ø Volumen (letzte 4 W) [kg·Wdh]"].round(1)
        out["Bestes e1RM (gesamter Zeitraum) [kg]"] = out["Bestes e1RM (gesamter Zeitraum) [kg]"].round(1)
        out["Trend e1RM (letzte 8 W) [kg/Woche]"] = out["Trend e1RM (letzte 8 W) [kg/Woche]"].map(lambda v: None if pd.isna(v) else round(v, 3))
        if {"Plateau (6 W)", "Bestes e1RM (gesamter Zeitraum) [kg]"}.issubset(out.columns):
            out = out.sort_values(["Plateau (6 W)", "Bestes e1RM (gesamter Zeitraum) [kg]"], ascending=[True, False])
    return out

# ----------------- Muskelgruppen: Mapping & Aggregation -----------------

DEFAULT_MUSCLE_GROUPS = [
    "Quadrizeps", "Hamstrings", "Gluteus", "Waden",
    "Rücken (oberer)", "Rücken (unterer)", "Latissimus",
    "Brust", "Schultern", "Bizeps", "Trizeps", "Core",
    "Ganzkörper", "Sonstiges"
]

def load_muscle_mapping(path: str) -> Dict[str, str]:
    """
    Lädt eine CSV mit Spalten: exercise, muscle_group (oder Übung, Muskelgruppe).
    Gibt ein Dict {exercise: muscle_group} zurück. Case-sensitiv pro Übungsname
    so wie in den Trainingsdaten.
    """
    try_enc = ["utf-8", "cp1252", "latin1"]
    for enc in try_enc:
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except Exception:
            df = None
    if df is None or df.empty:
        return {}
    # Spaltennamen tolerant erkennen
    cols = {c.lower(): c for c in df.columns}
    ex_col = cols.get("exercise") or cols.get("übung") or cols.get("uebung")
    mg_col = cols.get("muscle_group") or cols.get("muskelgruppe")
    if not ex_col or not mg_col:
        return {}
    mapping = {}
    for _, r in df.iterrows():
        ex = str(r[ex_col]).strip()
        mg = str(r[mg_col]).strip()
        if ex and mg:
            mapping[ex] = mg
    return mapping

def save_muscle_mapping(df_map: pd.DataFrame, path: str):
    """
    Speichert ein Mapping-DataFrame mit Spalten ['exercise','muscle_group'] als UTF-8 CSV.
    """
    out = df_map.rename(columns={"Übung": "exercise", "Muskelgruppe": "muscle_group"})
    out.to_csv(path, index=False, encoding="utf-8")

def build_weekly_by_muscle(weekly: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Aggregiert weekly (week, exercise, volume_sum, ...) -> pro Muskelgruppe.
    Übungen ohne Mapping gehen in 'Sonstiges'.
    """
    if weekly is None or weekly.empty:
        return weekly
    if mapping is None:
        mapping = {}
    df = weekly.copy()
    df["muscle_group"] = df["exercise"].map(mapping).fillna("Sonstiges")
    agg = (df.groupby(["week", "muscle_group"])
             .agg(volume_sum=("volume_sum", "sum"))
             .reset_index())
    return agg

# ----------------- Visualisierung -----------------

def _format_date_axis(ax):
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
    for label in ax.get_xticklabels():
        label.set_rotation(0)
        label.set_ha('center')

def plot_weekly_volume(weekly: pd.DataFrame, out_path: str = "weekly_volume.png") -> str | None:
    """Gesamtes Volumen je Übung – Legende außerhalb, keine Überlagerung."""
    if weekly.empty:
        return None
    pivot = weekly.pivot_table(index="week", columns="exercise", values="volume_sum", aggfunc="sum").fillna(0.0)
    fig = plt.figure(figsize=(10, 5))
    ax = plt.gca()
    pivot.plot(ax=ax)
    _format_date_axis(ax)
    ax.set_title("Wöchentliches Volumen pro Übung")
    ax.set_xlabel("Woche")
    ax.set_ylabel("Volumen (kg × Wiederholungen)")
    ax.legend(title="Übung", bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)
    plt.tight_layout(rect=[0,0,0.8,1])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path

def plot_weekly_volume_by_muscle(weekly_by_muscle: pd.DataFrame, out_path: str = "weekly_muscle_volume.png") -> str | None:
    """Gesamtes Volumen je Muskelgruppe – Legende außerhalb."""
    if weekly_by_muscle is None or weekly_by_muscle.empty:
        return None
    pivot = weekly_by_muscle.pivot_table(index="week", columns="muscle_group", values="volume_sum", aggfunc="sum").fillna(0.0)
    fig = plt.figure(figsize=(10, 5))
    ax = plt.gca()
    pivot.plot(ax=ax)
    _format_date_axis(ax)
    ax.set_title("Wöchentliches Volumen pro Muskelgruppe")
    ax.set_xlabel("Woche")
    ax.set_ylabel("Volumen (kg × Wiederholungen)")
    ax.legend(title="Muskelgruppe", bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)
    plt.tight_layout(rect=[0,0,0.8,1])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path

def plot_exercise_volume_bar(weekly: pd.DataFrame, exercise: str, out_path: str) -> str | None:
    """Balkendiagramm Volumen einer Übung je Woche."""
    w = weekly[weekly["exercise"]==exercise].sort_values("week")
    if w.empty: return None
    fig = plt.figure(figsize=(9,4.8))
    ax = plt.gca()
    ax.bar(w["week"], w["volume_sum"])
    _format_date_axis(ax)
    ax.set_title(f"{exercise}: Wöchentliches Volumen")
    ax.set_xlabel("Woche")
    ax.set_ylabel("Volumen (kg × Wiederholungen)")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150); plt.close(fig)
    return out_path

def plot_exercise_e1rm_line(weekly: pd.DataFrame, exercise: str, out_path: str, roll:int=3) -> str | None:
    """Linienplot bestes e1RM je Woche inkl. gleitendem Mittel."""
    w = weekly[weekly["exercise"]==exercise].sort_values("week")
    if w.empty: return None
    fig = plt.figure(figsize=(9,4.8))
    ax = plt.gca()
    ax.plot(w["week"], w["best_e1rm"], marker="o")
    if len(w) >= roll:
        ax.plot(w["week"], w["best_e1rm"].rolling(roll).mean(), linestyle="--")
    _format_date_axis(ax)
    ax.set_title(f"{exercise}: Bestes e1RM (Trend)")
    ax.set_xlabel("Woche")
    ax.set_ylabel("e1RM (kg – Epley)")
    ax.legend(["Bestes e1RM", f"Ø {roll} Wochen"], frameon=False, loc="upper left")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150); plt.close(fig)
    return out_path

def plot_exercise_max_weight(long: pd.DataFrame, exercise: str, out_path: str) -> str | None:
    """Maximales Trainingsgewicht pro Datum (Scatter)."""
    l = long[long["exercise"]==exercise].copy()
    if l.empty: return None
    day_max = l.groupby("date")["weight"].max().reset_index()
    fig = plt.figure(figsize=(9,4.8))
    ax = plt.gca()
    ax.scatter(day_max["date"], day_max["weight"], s=28)
    _format_date_axis(ax)
    ax.set_title(f"{exercise}: Maximales verwendetes Gewicht (täglich)")
    ax.set_xlabel("Datum")
    ax.set_ylabel("Gewicht (kg)")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150); plt.close(fig)
    return out_path

# ----------------- PR-Report -----------------

def pr_report(long: pd.DataFrame) -> pd.DataFrame:
    """Persönliche Rekorde pro Übung (max Gewicht & max e1RM)."""
    if long.empty:
        return long
    pr = (
        long.groupby("exercise")
            .agg(
                max_weight=("weight","max"),
                max_e1rm=("e1rm_epley","max")
            )
            .reset_index()
            .sort_values("max_e1rm", ascending=False)
    )
    return pr