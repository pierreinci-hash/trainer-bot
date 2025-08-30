# src/app.py
import os
import streamlit as st
from pathlib import Path
from typing import List, Tuple
import pandas as pd

from langchain_openai import ChatOpenAI

from settings import (
    OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL, OPENAI_EMBEDDING_MODEL,
    PERSIST_DIR, PDF_DIR, LOGS_DIR
)
from embeds import make_embeddings_with_fallback
from ingest import collect_pdfs, load_and_split, build_vectorstore, load_vectorstore_any
from trainings import (
    load_training_csv, to_long_form, weekly_summary, summarize_for_prompt,
    pr_report, plot_weekly_volume, plot_weekly_volume_by_muscle,
    detect_plateau, acwr, per_exercise_summary,
    plot_exercise_volume_bar, plot_exercise_e1rm_line, plot_exercise_max_weight,
    load_muscle_mapping, save_muscle_mapping, build_weekly_by_muscle, DEFAULT_MUSCLE_GROUPS
)
from prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

# --------- Helpers ---------
def _safe_file(name: str) -> str:
    return "".join(ch for ch in name if ch.isalnum() or ch in ("_", "-")).strip() or "plot"

def init_embeddings():
    embs, used_local = make_embeddings_with_fallback(OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL)
    if used_local:
        st.warning("OpenAI-Embeddings nicht erreichbar – lokaler Fallback (all-MiniLM-L6-v2) aktiv.", icon="⚠️")
    return embs

def init_llm(model_name: str):
    return ChatOpenAI(api_key=OPENAI_API_KEY, model=model_name, temperature=0.2)

def docs_block_from_retrieval(docs: List, max_chars: int = 12000) -> str:
    parts = []
    total = 0
    for i, d in enumerate(docs, start=1):
        source = d.metadata.get('source', 'Unbekannt')
        page = d.metadata.get('page', 'n/a')
        chunk = d.page_content.strip()
        entry = f"[{i}] ({source}, Seite {page})\n{chunk}"
        total += len(entry)
        if total > max_chars:
            break
        parts.append(entry)
    return "\n\n".join(parts)

def _date_bounds_from_weekly(weekly) -> Tuple[pd.Timestamp, pd.Timestamp]:
    if weekly is None or weekly.empty:
        today = pd.Timestamp.today().normalize()
        return today, today
    return weekly["week"].min().normalize(), weekly["week"].max().normalize()

def _apply_date_filter(long, weekly, start: pd.Timestamp, end: pd.Timestamp):
    if long is None or weekly is None or long.empty or weekly.empty:
        return long, weekly
    mask_long = (long["date"] >= start) & (long["date"] <= end + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
    long_f = long.loc[mask_long].copy()
    mask_week = (weekly["week"] >= start) & (weekly["week"] <= end)
    weekly_f = weekly.loc[mask_week].copy()
    return long_f, weekly_f

# --------- Streamlit UI ---------
st.set_page_config(page_title="AI Personal Trainer", layout="wide")
st.title("🏋️‍♂️ AI Personal Trainer – RAG + Trainingsanalyse")

MUSCLE_MAP_PATH = Path("data/muscles.csv")  # Mapping-CSV

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.caption("OpenAI-Zugang (aus Secrets/.env geladen)")
    st.text_input("OpenAI API Key", value=OPENAI_API_KEY, type="password", disabled=True)
    st.text_input("Base URL (bei OpenAI leer lassen)", value=OPENAI_BASE_URL or "", disabled=True)

    st.divider()
    st.subheader("Modelle")
    suggested_models = ["gpt-4o-mini","gpt-4o-mini-2024-07-18","gpt-4o","gpt-4.1-mini"]
    model_name = st.text_input("LLM-Modell (Chat)", value=OPENAI_MODEL or "gpt-4o-mini", help=f"Beispiele: {', '.join(suggested_models)}")
    st.text_input("Embedding-Modell (nur Info)", value=OPENAI_EMBEDDING_MODEL, disabled=True)

    st.divider()
    st.subheader("📚 Studien (PDF)")
    uploaded_pdfs = st.file_uploader("PDF(s) hochladen", type=["pdf"], accept_multiple_files=True)
    if uploaded_pdfs:
        for up in uploaded_pdfs:
            out = Path(PDF_DIR) / up.name
            with open(out, "wb") as f:
                f.write(up.getbuffer())
        st.success(f"{len(uploaded_pdfs)} PDF(s) gespeichert in data/pdfs")

    if st.button("PDFs indexieren"):
        if not OPENAI_API_KEY:
            st.error("Bitte zuerst OPENAI_API_KEY setzen (Secrets/.env).")
        else:
            with st.spinner("Lese PDFs ein und erstelle Vektorindex ..."):
                pdfs = collect_pdfs(PDF_DIR)
                if not pdfs:
                    st.warning("Keine PDFs im Ordner 'data/pdfs' gefunden.")
                else:
                    docs = load_and_split(pdfs)
                    embs = init_embeddings()
                    backend, n = build_vectorstore(docs, embeddings=embs)
                    if n == 0:
                        st.warning("Es konnten keine Text-Chunks erzeugt werden.")
                    else:
                        st.success(f"Fertig! {n} Chunks indexiert. Backend: **{backend.upper()}**")

    st.subheader("📈 Trainingsdaten (CSV)")
    csv_files = [f for f in Path(LOGS_DIR).glob("*.csv")]
    choice = st.selectbox("CSV-Datei auswählen", ["(keine)"] + [f.name for f in csv_files])
    uploaded_csv = st.file_uploader("Oder CSV hochladen", type=["csv"])
    if uploaded_csv is not None:
        tmp_path = Path(LOGS_DIR) / uploaded_csv.name
        with open(tmp_path, "wb") as tmp:
            tmp.write(uploaded_csv.getbuffer())
        st.success(f"Gespeichert: {tmp_path.name}")
        choice = tmp_path.name

    if 'train_long' not in st.session_state:
        st.session_state['train_long'] = None
        st.session_state['weekly'] = None
        st.session_state['train_summary'] = "Keine Trainingsdaten geladen."
        st.session_state['filter_mode'] = "Gesamte Zeit"
        st.session_state['filter_start'] = None
        st.session_state['filter_end'] = None
        st.session_state['muscle_map_df'] = None  # Editor-DF

    if st.button("CSV laden/aktualisieren"):
        target = None
        if choice and choice != "(keine)":
            target = Path(LOGS_DIR) / choice
        if target and target.exists():
            df_raw = load_training_csv(str(target))
            long = to_long_form(df_raw)
            weekly = weekly_summary(long)

            st.session_state['train_long'] = long
            st.session_state['weekly'] = weekly
            st.session_state['train_summary'] = summarize_for_prompt(weekly)

            exercises = sorted(weekly["exercise"].unique().tolist()) if weekly is not None and not weekly.empty else []
            existing_map = load_muscle_mapping(str(MUSCLE_MAP_PATH)) if MUSCLE_MAP_PATH.exists() else {}
            st.session_state['muscle_map_df'] = pd.DataFrame(
                [{"Übung": ex, "Muskelgruppe": existing_map.get(ex, "Sonstiges")} for ex in exercises],
                columns=["Übung", "Muskelgruppe"]
            )

            if st.session_state['muscle_map_df'] is not None and not st.session_state['muscle_map_df'].empty:
                tmp_map = dict(zip(st.session_state['muscle_map_df']["Übung"], st.session_state['muscle_map_df']["Muskelgruppe"]))
                weekly_m = build_weekly_by_muscle(weekly, tmp_map)
                img_path = plot_weekly_volume_by_muscle(weekly_m, out_path=str(Path("weekly_muscle_volume.png")))
                if img_path:
                    st.image(img_path, caption="Wöchentliches Volumen pro Muskelgruppe (gesamter Zeitraum)", use_column_width=True)
            else:
                img_path = plot_weekly_volume(weekly, out_path=str(Path("weekly_volume.png")))
                if img_path:
                    st.image(img_path, caption="Wöchentliches Volumen (pro Übung)", use_column_width=True)

            col2a, col2b = st.columns(2)
            with col2a:
                n_sets = int(long.shape[0]) if long is not None and not long.empty else 0
                n_ex = int(weekly["exercise"].nunique()) if weekly is not None and not weekly.empty else 0
                st.markdown("**Status Trainingsdaten (gesamt)**")
                st.write(f"- Erkannte Sätze: **{n_sets}**")
                st.write(f"- Übungen (distinct): **{n_ex}**")
            with col2b:
                ratio = acwr(long)
                if ratio is not None:
                    st.info(f"ACWR (7d/28d) – gesamt: **{ratio:.2f}**  – Zielbereich oft ~0.8–1.3", icon="ℹ️")

            st.success("Trainingsdaten geladen.")
        else:
            st.warning("Bitte eine gültige CSV auswählen oder hochladen.")

# --------- Muskelgruppen-Mapping (Editor) ---------
long = st.session_state.get('train_long')
weekly = st.session_state.get('weekly')
st.divider()
st.subheader("🧩 Muskelgruppen-Mapping (Übung → Muskelgruppe)")

if weekly is not None and not weekly.empty:
    if st.session_state.get('muscle_map_df') is None or st.session_state['muscle_map_df'].empty:
        exercises = sorted(weekly["exercise"].unique().tolist())
        st.session_state['muscle_map_df'] = pd.DataFrame({"Übung": exercises, "Muskelgruppe": ["Sonstiges"]*len(exercises)})

    options = DEFAULT_MUSCLE_GROUPS
    edited = st.data_editor(
        st.session_state['muscle_map_df'],
        num_rows="dynamic",
        column_config={
            "Übung": st.column_config.TextColumn(disabled=True),
            "Muskelgruppe": st.column_config.SelectboxColumn(options=options)
        },
        use_container_width=True,
        key="editor_muscles"
    )

    col_s1, col_s2 = st.columns([1,1])
    with col_s1:
        if st.button("💾 Mapping speichern"):
            st.session_state['muscle_map_df'] = edited.copy()
            save_muscle_mapping(st.session_state['muscle_map_df'], str(MUSCLE_MAP_PATH))
            st.success(f"Gespeichert: {MUSCLE_MAP_PATH}")
    with col_s2:
        if st.button("📥 Mapping aus Datei laden"):
            if MUSCLE_MAP_PATH.exists():
                m = load_muscle_mapping(str(MUSCLE_MAP_PATH))
                exercises = sorted(weekly["exercise"].unique().tolist())
                rows = [{"Übung": ex, "Muskelgruppe": m.get(ex, "Sonstiges")} for ex in exercises]
                st.session_state['muscle_map_df'] = pd.DataFrame(rows, columns=["Übung", "Muskelgruppe"])
                st.success("Mapping geladen.")
            else:
                st.warning("Es existiert noch keine Mapping-Datei unter data/muscles.csv.")

# --------- Zeitraum-Filter ---------
if long is not None and not long.empty and weekly is not None and not weekly.empty:
    st.divider()
    st.subheader("🗓️ Zeitraum-Filter für Analysen & Plots")

    min_w, max_w = _date_bounds_from_weekly(weekly)
    mode = st.radio("Zeitraum wählen", ["Gesamte Zeit", "Letzte N Wochen", "Benutzerdefiniert"], horizontal=True, index=0)

    if mode == "Gesamte Zeit":
        start, end = min_w, max_w
    elif mode == "Letzte N Wochen":
        n = st.slider("Anzahl Wochen", min_value=4, max_value=52, value=12, step=1)
        end = max_w
        start = (end - pd.Timedelta(weeks=n-1)).normalize()
    else:
        default = (min_w.date(), max_w.date())
        sel = st.date_input("Zeitraum (Von/Bis)", value=default)
        if isinstance(sel, tuple) and len(sel) == 2:
            start = pd.Timestamp(sel[0])
            end = pd.Timestamp(sel[1])
        else:
            start, end = min_w, max_w

    st.session_state['filter_mode'] = mode
    st.session_state['filter_start'] = start
    st.session_state['filter_end'] = end

    long_f, weekly_f = _apply_date_filter(long, weekly, start, end)

    # Gefiltertes Muskelgruppen-Diagramm (falls Mapping existiert)
    if st.session_state.get('muscle_map_df') is not None and not st.session_state['muscle_map_df'].empty:
        tmp_map = dict(zip(st.session_state['muscle_map_df']["Übung"], st.session_state['muscle_map_df']["Muskelgruppe"]))
        weekly_m = build_weekly_by_muscle(weekly_f, tmp_map)
        p = plot_weekly_volume_by_muscle(weekly_m, out_path=f"weekly_muscle_{start.date()}_{end.date()}.png")
        if p:
            st.image(p, caption=f"Wöchentliches Volumen pro Muskelgruppe [{start.date()} bis {end.date()}]", use_column_width=True)

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        st.write(f"**Aktiver Zeitraum:** {start.date()} bis {end.date()}")
        st.write(f"- Sätze im Zeitraum: **{int(long_f.shape[0])}**")
        st.write(f"- Wochen im Zeitraum: **{int(weekly_f['week'].nunique())}**")
    with col_f2:
        summary_df = per_exercise_summary(long_f, weekly_f)
        st.dataframe(summary_df, use_container_width=True)

    st.subheader("🎨 Visualisierung & Analyse (gezielt)")
    if not weekly_f.empty:
        exercises = sorted(weekly_f["exercise"].unique().tolist())
        ex = st.selectbox("Übung auswählen", exercises, key="ex_select_filtered")
        colA, colB, colC = st.columns(3)
        with colA:
            show_vol = st.checkbox("Volumen je Woche (Balken)", value=True, key="ch_vol")
        with colB:
            show_e1  = st.checkbox("Bestes e1RM (Linie)", value=True, key="ch_e1")
        with colC:
            show_max = st.checkbox("Max. Gewicht pro Tag (Scatter)", value=False, key="ch_max")

        if st.button("Diagramme erzeugen", key="btn_plots_filtered"):
            if show_vol:
                p = plot_exercise_volume_bar(weekly_f, ex, out_path=f"{_safe_file(ex)}_vol.png")
                if p: st.image(p, use_column_width=True)
            if show_e1:
                p = plot_exercise_e1rm_line(weekly_f, ex, out_path=f"{_safe_file(ex)}_e1rm.png")
                if p: st.image(p, use_column_width=True)
            if show_max:
                p = plot_exercise_max_weight(long_f, ex, out_path=f"{_safe_file(ex)}_maxw.png")
                if p: st.image(p, use_column_width=True)

            plat = detect_plateau(weekly_f, ex, weeks=6)
            if plat.get("enough_data"):
                verdict = "⚠️ Plateau-Verdacht" if plat["is_plateau"] else "✅ Positiver Trend"
                st.write(f"**Trend (6W, gefiltert):** Slope={plat['slope_per_week']:.2f} → {verdict}")
            else:
                st.write("Für die Plateau-Analyse liegen im gewählten Zeitraum zu wenige Wochen vor.")

# --------- Chat ---------
st.divider()
st.write("### Chat")
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

for role, content in st.session_state['messages']:
    with st.chat_message(role):
        st.markdown(content)

user_input = st.chat_input("Frag mich etwas (z. B. 'Welche Muskelgruppe ist untertrainiert?' oder 'plot: Kniebeuge e1rm')")

if user_input:
    if not OPENAI_API_KEY:
        st.error("Bitte zuerst OPENAI_API_KEY setzen (Secrets/.env).")
        st.stop()

    # Shortcut: "plot:" nutzt den aktiven Zeitraum
    if user_input.lower().startswith("plot:"):
        long = st.session_state.get('train_long'); weekly = st.session_state.get('weekly')
        if long is None or weekly is None or long.empty or weekly.empty:
            st.warning("Bitte lade zunächst eine CSV (Trainingsdaten).")
        else:
            start = st.session_state.get('filter_start') or _date_bounds_from_weekly(weekly)[0]
            end   = st.session_state.get('filter_end') or _date_bounds_from_weekly(weekly)[1]
            long_f, weekly_f = _apply_date_filter(long, weekly, start, end)
            try:
                cmd = user_input[5:].strip()
                parts = cmd.split()
                metric = "vol"
                if len(parts) >= 2 and parts[-1].lower() in ("vol", "e1rm", "max"):
                    metric = parts[-1].lower()
                    ex_name = " ".join(parts[:-1])
                else:
                    ex_name = cmd
                ex_name = ex_name.strip()
                if ex_name not in weekly_f["exercise"].unique():
                    st.warning(f"Übung im gewählten Zeitraum nicht gefunden: '{ex_name}'. Verfügbare: {', '.join(sorted(weekly_f['exercise'].unique()))}")
                else:
                    if metric == "vol":
                        p = plot_exercise_volume_bar(weekly_f, ex_name, out_path=f"{_safe_file(ex_name)}_vol.png")
                    elif metric == "e1rm":
                        p = plot_exercise_e1rm_line(weekly_f, ex_name, out_path=f"{_safe_file(ex_name)}_e1rm.png")
                    else:
                        p = plot_exercise_max_weight(long_f, ex_name, out_path=f"{_safe_file(ex_name)}_maxw.png")
                    if p:
                        with st.chat_message("assistant"):
                            st.image(p, caption=f"{ex_name} – Diagramm ({metric}) [{start.date()} bis {end.date()}]", use_column_width=True)
                    st.session_state['messages'].append(("user", user_input))
                    st.session_state['messages'].append(("assistant", f"[Diagramm erstellt: {ex_name} – {metric} – Zeitraum {start.date()} bis {end.date()}]"))
            except Exception as e:
                st.error(f"Plot-Fehler: {e}")
        st.stop()

    # RAG-Chat
    long = st.session_state.get('train_long'); weekly = st.session_state.get('weekly')
    start = st.session_state.get('filter_start') or _date_bounds_from_weekly(weekly)[0] if weekly is not None else pd.Timestamp.today()
    end   = st.session_state.get('filter_end') or _date_bounds_from_weekly(weekly)[1] if weekly is not None else pd.Timestamp.today()

    long_f, weekly_f = _apply_date_filter(long, weekly, start, end) if (long is not None and weekly is not None) else (None, None)

    embs = make_embeddings_with_fallback(OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL)[0]
    llm = ChatOpenAI(api_key=OPENAI_API_KEY, model=OPENAI_MODEL or "gpt-4o-mini", temperature=0.2)

    # <- NEU: Index laden mit Backend-Erkennung
    vs = load_vectorstore_any(embs)
    retrieved = []
    if vs is not None:
        retriever = vs.as_retriever(search_kwargs={'k': 4})
        try:
            retrieved = retriever.get_relevant_documents(user_input)
        except Exception as e:
            st.warning(f"Retrieval fehlgeschlagen: {e}")

    docs_block = docs_block_from_retrieval(retrieved) if retrieved else "Keine Studienpassagen gefunden."
    train_context = summarize_for_prompt(weekly_f) if (weekly_f is not None and not weekly_f.empty) else "Keine Trainingsdaten im gewählten Zeitraum."

    from langchain.schema import SystemMessage, HumanMessage
    user_prompt = USER_PROMPT_TEMPLATE.format(question=user_input, train_context=train_context, docs_block=docs_block)
    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_prompt)]

    with st.spinner("Denke nach ..."):
        try:
            resp = llm.invoke(messages)
            answer = resp.content
        except Exception as e:
            answer = (
                "Fehler beim LLM-Aufruf:\n\n"
                f"```\n{e}\n```\n\n"
                "Falls dein Netzwerk `api.openai.com` blockiert, nutze mobilen Hotspot/VPN."
            )

    st.session_state['messages'].append(("user", user_input))
    st.session_state['messages'].append(("assistant", answer))

    with st.chat_message("assistant"):
        st.markdown(answer)

    if retrieved:
        with st.expander("Verwendete Quellen / Passagen"):
            for d in retrieved:
                st.markdown(f"- **{d.metadata.get('source','?')}**, Seite {d.metadata.get('page','?')}")