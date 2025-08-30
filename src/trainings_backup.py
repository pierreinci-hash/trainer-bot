from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import unicodedata
from typing import Optional

# ---------- Hilfsfunktionen für robuste CSV-Verarbeitung ----------

def _to_float(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    x = str(x).strip().replace(',', '.')
    try:
        return float(x)
    except:
        return np.nan

def _normalize_ascii(s: str) -> str:
    """
    Entfernt Umlaute/Akzente (Ü->Ue, ä->ae etc. via ASCII-Approx), 
    kleinschreiben, Leerzeichen/sonderzeichen raus.
    """
    if s is None:
        return ""
    # Erst NFKD -> ASCII (diacritics weg)
    no_diac = (
        unicodedata.normalize('NFKD', str(s))
        .encode('ascii', 'ignore')
        .decode('ascii')
    )
    # Kleinschreiben, Leer/sonderzeichen vereinheitlichen
    clean = ''.join(ch for ch in no_diac if ch.isalnum())
    return clean.lower()

def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mappt Originalspalten auf eine standardisierte, umlautfreie Form.
    Erwartet mindestens 'datum' und 'uebung' (oder Varianten).
    Satz/Reps-Paare: satz1, wdh1, satz2, wdh2, ...
    """
    mapping = {}
    for col in df.columns:
        norm = _normalize_ascii(col)  # z.B. "Übung" -> "ubung"
        # Sonderfall: "Uebung" vs "Ubung"
        if norm == "ubung":
            std = "uebung"
        elif norm == "datum":
            std = "datum"
        elif norm.startswith("satz") and norm[4:].isdigit():
            std = f"satz{norm[4:]}"
        elif norm.startswith("wdh") and norm[3:].isdigit():
            std = f"wdh{norm[3:]}"
        else:
            # sonst einfach den normalisierten Namen nehmen
            std = norm
        mapping[col] = std

    df = df.rename(columns=mapping)
    return df

# ---------- CSV laden & in Long-Form bringen ----------

def load_training_csv(path: str) -> pd.DataFrame:
    """
    Lädt das CSV robust:
    1) Versuch UTF-8, 2) Fallback cp1252, 3) Fallback latin1.
    Erkennt Trennzeichen automatisch.
    """
    encodings = ["utf-8", "cp1252", "latin1"]
    last_err = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc, sep=None, engine='python')
            break
        except Exception as e:
            last_err = e
            df = None
    if df is None:
        raise last_err

    # Spalten standardisieren (Umlaute entfernen, einheitliche Keys)
    df = _standardize_columns(df)

    # Datum parsen (deutsches Format robust)
    if "datum" not in df.columns:
        raise ValueError("Spalte 'Datum' (oder Variante) nicht gefunden.")
    df["datum"] = pd.to_datetime(df["datum"], dayfirst=True, errors='coerce')
    df = df.dropna(subset=["datum"])

    # Übungsname
    # Akzeptiere sowohl 'uebung' als auch (zur Sicherheit) 'ubung'
    exercise_col = "uebung" if "uebung" in df.columns else ("ubung" if "ubung" in df.columns else None)
    if exercise_col is None:
        raise ValueError("Spalte 'Übung'/'Uebung' nicht gefunden.")

    # Sortiere Spalten neu an (optional)
    ordered = ["datum", exercise_col] + [c for c in df.columns if c not in ("datum", exercise_col)]
    df = df[ordered]
    return df

def to_long_form(df: pd.DataFrame) -> pd.DataFrame:
    """
    Erwartet Spalten:
      - datum, uebung (oder ubung)
      - satz1, wdh1, satz2, wdh2, ...
    """
    exercise_col = "uebung" if "uebung" in df.columns else "ubung"
    base_cols = ["datum", exercise_col]
    set_cols = [c for c in df.columns if c not in base_cols]

    # Paare bilden (satzN, wdhN)
    pairs = []
    for c in set_cols:
        if c.startswith("satz") and c[4:].isdigit():
            idx = int(c[4:])
            reps_col = f"wdh{idx}"
            if reps_col in df.columns:
                pairs.append((c, reps_col, idx))

    rows = []
    for _, row in df.iterrows():
        for weight_col, reps_col, idx in pairs:
            w = _to_float(row.get(weight_col))
            r = _to_float(row.get(reps_col))
            if np.isnan(w) or np.isnan(r):
                continue
            rows.append({
                'date': row['datum'].date(),
                'exercise': str(row[exercise_col]).strip(),
                'set_idx': idx,
                'weight': w,
                'reps': r,
                'volume': w * r,
                'e1rm_epley': w * (1 + r/30.0),
            })
    long = pd.DataFrame(rows)
    if long.empty:
        return long
    long['date'] = pd.to_datetime(long['date'])
    long['week'] = long['date'] - long['date'].dt.weekday * pd.Timedelta(days=1)
    return long

# ---------- Auswertungen ----------

def weekly_summary(long: pd.DataFrame) -> pd.DataFrame:
    if long.empty:
        return long
    agg = (long
           .groupby(['week', 'exercise'])
           .agg(volume_sum=('volume','sum'),
                hard_sets=('set_idx','count'),
                best_e1rm=('e1rm_epley','max'))
           .reset_index())
    return agg

def detect_plateau(weekly: pd.DataFrame, exercise: str, weeks:int=6) -> dict:
    data = weekly[weekly['exercise']==exercise].sort_values('week').tail(weeks)
    if len(data) < max(3, weeks//2):
        return {'enough_data': False}
    X = np.arange(len(data)).reshape(-1,1)
    y = data['best_e1rm'].values
    model = LinearRegression().fit(X,y)
    slope = float(model.coef_[0])
    return {
        'enough_data': True,
        'slope_per_week': slope,
        'is_plateau': abs(slope) < 0.25  # ~ <0.25 kg pro Woche
    }

def acwr(long: pd.DataFrame, day_window_acute:int=7, day_window_chronic:int=28) -> Optional[float]:
    if long.empty:
        return None
    df = long[['date','volume']].copy()
    df = df.groupby('date').sum().asfreq('D').fillna(0.0)
    acute = df.tail(day_window_acute)['volume'].sum()
    chronic = df.tail(day_window_chronic)['volume'].mean() if len(df)>=day_window_chronic else None
    if not chronic or chronic == 0:
        return None
    return float(acute / chronic)

def summarize_for_prompt(weekly: pd.DataFrame, top_n:int=8) -> str:
    if weekly.empty:
        return "Keine Trainingsdaten geladen."
    last_weeks = weekly['week'].max()
    if pd.isna(last_weeks):
        return "Keine Trainingsdaten geladen."
    parts = []
    for ex, grp in weekly.groupby('exercise'):
        g = grp.sort_values('week').tail(8)
        if g.empty:
            continue
        line = f"- {ex}: Volumen/Woche {[round(v,1) for v in g['volume_sum'].tolist()]}; beste e1RM {[round(v,1) for v in g['best_e1rm'].tolist()]}"
        parts.append(line)
    if not parts:
        return "Keine verwertbaren Sätze gefunden."
    return "\n".join(parts)

def pr_report(long: pd.DataFrame) -> pd.DataFrame:
    if long.empty:
        return long
    pr = (long.groupby('exercise')
                .agg(max_weight=('weight','max'),
                     max_e1rm=('e1rm_epley','max'))
                .reset_index())
    return pr

def plot_weekly_volume(weekly: pd.DataFrame, out_path: str = "weekly_volume.png") -> str | None:
    if weekly.empty:
        return None
    import matplotlib.pyplot as plt
    pivot = weekly.pivot_table(index='week', columns='exercise', values='volume_sum', aggfunc='sum').fillna(0.0)
    fig = plt.figure()
    pivot.plot(ax=plt.gca())  # keine spezifischen Farben setzen
    plt.title("Wöchentliches Volumen pro Übung")
    plt.xlabel("Woche")
    plt.ylabel("Volumen (kg x Wiederholungen)")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path