import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit.components.v1 as components
from st_screen_stats import ScreenData
from streamlit_sortables import sort_items
import math
import scipy.stats as stats
import pymannkendall as mk

# Funktion, die überprüft, ob die Suchparameter nur Trainingseinheiten enthalten bzw. ob SpielerPlus Filter richtig gesetzt sind
def check_training_only(param_str: str) -> bool:
    parts = [p.strip() for p in param_str.split("|")]
    flags = {p.split(":")[0].strip(): p.split(":")[1].strip()
             for p in parts if not p.strip().startswith("search_date")}
    
    return flags == {
        "training": "YES",
        "game": "NO",
        "tournament": "NO",
        "event": "NO"
    }

def check_search_params(raw_data):
    # Wendet den Test auf alle Zeilen an
    result = raw_data['search_params'].apply(check_training_only)
    # Prüfen, ob es für das ganze DF gilt
    all_ok = result.all()
    return bool(all_ok)

def filter_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    # Unnötige Spalten entfernen
    raw_data = raw_data.drop(['event_id', 'team_id', 'team_name', 'event_name', 'user_id', 'search_params', 'event_type'], axis=1)
    
    # event_date_end entfernen
    raw_data = raw_data.drop(['event_date_end'], axis=1)
    
    # event_date_start in datetime umwandeln
    raw_data['event_date_start'] = pd.to_datetime(raw_data['event_date_start'], format='%d-%m-%Y')
    
    # Filtern der Daten, um nur nominierte Spieler zu behalten
    df_filtered = raw_data[raw_data['user_participation'] != 'STATUS_NOT_NOMINATED']
    
    return df_filtered

# dict für start daten
player_start_dates = {
    'Jarne Blanke': datetime(2025, 5, 22),
    'Lukas Welzel': datetime(2025, 6, 13),
    'Julian M.': datetime(2025, 5, 16),
    'Jasper Kallmeyer': datetime(2025, 2, 20),
    'Clemens Galonsky': datetime(2025, 5, 16),
    'Jo Mathis': datetime(2025, 5, 16),
    'Mattis Bertuleit': datetime(2025, 2, 27),
    'Merlin Luc': datetime(2025, 1, 29),
}

# Grundwerte individuelle Spieler
def grundwerte(df, start_date=None, end_date=None):
    # Optional: Zeitraum-Filter
    if start_date and end_date:
        df = df[(df['event_date_start'].dt.date >= start_date) & 
                (df['event_date_start'].dt.date <= end_date)]
    
    # Basisindex: alle Spieler im Datensatz
    all_users = df['user_name'].unique()
    
    # Filter: Nur Events nach individuellem Startdatum
    df_filtered = df[
        df.apply(lambda row: row['event_date_start'] > player_start_dates.get(row['user_name'], datetime.min), axis=1)
    ]
    
    # Gesamtanzahl Sessions ab Startdatum
    total_sessions = df_filtered.groupby('user_name')['event_date_start'].nunique()
    
    # Abwesenheiten ab Startdatum
    absence_sessions = (
        df_filtered[df_filtered['user_participation'] == 'STATUS_ABSENCE']
        .groupby('user_name')['event_date_start']
        .nunique()
        .reindex(total_sessions.index, fill_value=0)
    )
    
    confirmed = (
        df_filtered[df_filtered['user_participation'] == 'STATUS_CONFIRMED']
        .groupby('user_name')['event_date_start']
        .nunique()
        .reindex(all_users, fill_value=0)
    )
    
    unsure = (
        df_filtered[df_filtered['user_participation'] == 'STATUS_UNSURE']
        .groupby('user_name')['event_date_start']
        .nunique()
        .reindex(all_users, fill_value=0)
    )
    
    not_choosed = (
        df_filtered[df_filtered['user_participation'] == 'STATUS_NOT_CHOOSED']
        .groupby('user_name')['event_date_start']
        .nunique()
        .reindex(total_sessions.index, fill_value=0)
    )
    
    eligible_sessions = total_sessions - absence_sessions
    
    result = (
        pd.DataFrame({
            "total_sessions": total_sessions,
            "absence_sessions": absence_sessions,
            "confirmed": confirmed,
            "unsure": unsure,
            "not_choosed": not_choosed,
            "eligible_sessions": eligible_sessions,
        })
        .reindex(all_users, fill_value=0)
    )
    
    return result

# Wilson Lower Bound Funktion für Verlässlichkeit
def wilson_lower_bound(pos, n, confidence=0.95):
    """
    Function to provide lower bound of wilson score
    :param pos: No of positive ratings
    :param n: Total number of ratings
    :param confidence: Confidence interval, by default is 95 %
    :return: Wilson Lower bound score
    """
    if n == 0:
        return 0
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * pos / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

def calculate_reliability(confirmed, eligible_sessions):
    # Berechnung des Wilson Lower Bound Scores für jeden Spieler
    reliability_score = confirmed.apply(lambda x: wilson_lower_bound(pos=x, n=eligible_sessions[confirmed.index[confirmed == x][0]]))
    return reliability_score

# Commitment Funktionen
def commitment(grundwerte_df):
    commitment_hard = grundwerte_df['confirmed'] / grundwerte_df['eligible_sessions']
    commitment_soft = (grundwerte_df['confirmed'] * 1 + grundwerte_df['unsure'] * 0.5 + grundwerte_df['not_choosed'] * 0.25) / grundwerte_df['eligible_sessions']
    return commitment_hard, commitment_soft

# Ewm Commitment bzw. aktuelles
def prepare_data_ewm(df, player_start_dates):
    df = df.copy()
    df['event_date_start'] = pd.to_datetime(df['event_date_start'])
    
    # Abwesenheiten & Ferien entfernen
    df = df[df['user_participation'] != 'STATUS_ABSENCE']
    
    # Spieler-Startdaten berücksichtigen
    df = df[
        df.apply(
            lambda row: row['event_date_start'] > player_start_dates.get(row['user_name'], datetime.min),
            axis=1
        )
    ]
    
    # Scores berechnen
    df['soft_score'] = df['user_participation'].map({
        'STATUS_CONFIRMED': 1.0,
        'STATUS_UNSURE': 0.5,
        'STATUS_NOT_CHOOSED': 0.25,
        'STATUS_REJECTED': 0.0
    })
    
    df['hard_score'] = df['user_participation'].map({
        'STATUS_CONFIRMED': 1.0,
        'STATUS_UNSURE': 0.0,
        'STATUS_NOT_CHOOSED': 0.0,
        'STATUS_REJECTED': 0.0
    })
    
    return df

def compute_ewm(group, halflife="14d"):
    group = group.sort_values("event_date_start")
    group["soft_commitment_ewm"] = (
        group["soft_score"].ewm(halflife=halflife, times=group["event_date_start"]).mean()
    )
    group["hard_commitment_ewm"] = (
        group["hard_score"].ewm(halflife=halflife, times=group["event_date_start"]).mean()
    )
    return group

def get_ewm_full(df, halflife="14d"):
    return df.groupby("user_name", group_keys=False).apply(lambda g: compute_ewm(g, halflife))

# Trendanalyse (hoch runter neutral)
def detect_trend(df, player_name, n_trainings=10, start_date=None, end_date=None):
    """
    Analyzes the trend of a player's hard commitment EWM scores using the Mann-Kendall test
    for the last N trainings and filters by p-value.
    """
    df_player = df[df['user_name'] == player_name]
    
    # Optional: Zeitfilter
    if start_date and end_date:
        df_player = df_player[
            (df_player['event_date_start'].dt.date >= start_date) &
            (df_player['event_date_start'].dt.date <= end_date)
        ]
    
    if len(df_player) < n_trainings:
        return None
    
    # Get the last n_trainings based on unique event dates
    last_n_dates = df_player.sort_values('event_date_start')['event_date_start'].unique()[-n_trainings:]
    df_player_last_n = df_player[df_player['event_date_start'].isin(last_n_dates)].sort_values('event_date_start')
    
    if len(df_player_last_n) < 3:  # Mann-Kendall requires at least 3 data points
        return None
    
    hard_result = mk.original_test(df_player_last_n['hard_commitment_ewm'], alpha=0.05)
    
    # Filter results based on p-value
    return {
        'player_name': player_name,
        'hard_trend': hard_result.trend,
        'slope': hard_result.slope,
        'intercept': hard_result.intercept
    }

def team_grundwerte(df):
    df_relevant_attendance_per_training = df[df['user_participation'] != 'STATUS_ABSENCE'].copy()
    df_hard_commitment_per_training = df[df['user_participation'] == 'STATUS_CONFIRMED'].copy()
    
    # Gesamtanzahl der relevanten Teilnehmer pro Training
    total_relevant_per_training = (
        df_relevant_attendance_per_training
        .groupby('event_date_start')
        .size()
        .reset_index(name='total_relevant')
    )
    
    # Anzahl der bestätigten Teilnehmer pro Training
    confirmed_per_training = (
        df_hard_commitment_per_training
        .groupby('event_date_start')
        .size()
        .reset_index(name='confirmed_attendances')
    )
    
    # Zusammenführen der beiden DataFrames
    training_attendance = pd.merge(
        total_relevant_per_training,
        confirmed_per_training,
        on='event_date_start',
        how='left'
    ).fillna(0)
    
    # Hard Commitment Rate pro Training berechnen
    training_attendance['hard_commitment_rate'] = (
        training_attendance['confirmed_attendances'] / training_attendance['total_relevant']
    ) * 100
    
    # Sortieren nach Datum
    training_attendance = training_attendance.sort_values('event_date_start')
    
    return training_attendance

def std_team(training_attendance):
    std = training_attendance['hard_commitment_rate'].std()
    return std

def avg_players_per_training(df):
    # Filter for confirmed attendances
    confirmed_df = df[df['user_participation'] == 'STATUS_CONFIRMED']
    
    # Calculate the average number of confirmed attendances per training
    average_confirmed_attendances = confirmed_df.groupby('event_date_start').size().mean()
    
    return average_confirmed_attendances

# Dataframe mit allen Sachen
def final_df(df_trends, ewm_full, grundwerte_df):
    df_trends1 = df_trends.copy()
    df_trends1.loc[df_trends1["hard_trend"] == "no trend", ["slope", "intercept"]] = 0
    df_trends1 = df_trends1.drop(columns=["hard_trend"])
    
    # === 2) EWM: nur letzter Wert je Spieler ===
    ewm_last = ewm_full.groupby("user_name").agg(
        hard_commitment_ewm=("hard_commitment_ewm", "last"),
        soft_commitment_ewm=("soft_commitment_ewm", "last")
    )
    
    # === 3) Grundwerte sind schon pro Spieler aggregiert ===
    grundwerte1 = grundwerte_df.copy()
    
    # === 4) Zusammenführen ===
    final_df = (
        grundwerte1
        .join(ewm_last, on="user_name")
        .reset_index()
        .merge(df_trends1.reset_index(), how="left", left_on="user_name", right_on="player_name")
    )
    
    # Index setzen
    final_df.set_index("user_name", inplace=True)
    final_df.drop(columns=["player_name"], inplace=True, errors='ignore')
    
    return final_df

# Visualisierungen
def team_commitment_chart(team_grundwerte_df, timeframe):
    df = team_grundwerte_df.copy()
    
    if timeframe == "Tage":
        # Original: Liniendiagramm pro Trainingstag
        fig = px.line(
            df,
            x="event_date_start",
            y="hard_commitment_rate",
            markers=True,
            title="Hard Commitment Rate pro Training (ohne Absence)",
            labels={
                "event_date_start": "Trainingsdatum",
                "hard_commitment_rate": "Hard Commitment Rate (%)"
            }
        )
        fig.update_yaxes(range=[0, 100])
        fig.update_layout(xaxis=dict(tickangle=45))
        fig.update_xaxes(range=[df['event_date_start'].min(), df['event_date_start'].max()])
    
    elif timeframe == "Wochen":
        # Gruppierung nach Kalenderwoche - BEHOBEN
        df["week"] = df["event_date_start"].dt.to_period("W").dt.start_time
        df_weekly = df.groupby("week")["hard_commitment_rate"].mean().reset_index()
        
        fig = px.bar(
            df_weekly,
            x="week",
            y="hard_commitment_rate",
            title="Durchschnittlicher Hard Commitment Rate pro Woche",
            labels={
                "week": "Kalenderwoche",
                "hard_commitment_rate": "Hard Commitment Rate (%)"
            }
        )
        fig.update_yaxes(range=[0, 100])
    
    elif timeframe == "Monate":
        # Gruppierung nach Monat - BEHOBEN
        df["month"] = df["event_date_start"].dt.to_period("M").dt.start_time
        df_monthly = df.groupby("month")["hard_commitment_rate"].mean().reset_index()
        
        fig = px.bar(
            df_monthly,
            x="month",
            y="hard_commitment_rate",
            title="Durchschnittlicher Hard Commitment Rate pro Monat",
            labels={
                "month": "Monat",
                "hard_commitment_rate": "Hard Commitment Rate (%)"
            }
        )
        fig.update_yaxes(range=[0, 100])
    
    return fig

# Pfeil
def arrow(team_grundwerte_df, time_team_commitment):
    df_steigung = team_grundwerte_df.copy()
    
    if time_team_commitment == "Wochen":
        df_steigung["week"] = df_steigung["event_date_start"].dt.to_period("W").dt.start_time
        df_steigung = df_steigung.groupby("week")["hard_commitment_rate"].mean().reset_index()
    elif time_team_commitment == "Monate":
        df_steigung["month"] = df_steigung["event_date_start"].dt.to_period("M").dt.start_time
        df_steigung = df_steigung.groupby("month")["hard_commitment_rate"].mean().reset_index()
    elif time_team_commitment == "Tage":
        df_steigung = team_grundwerte_df
    
    if len(df_steigung) < 2:
        return None, None
    
    arrow_value = df_steigung['hard_commitment_rate'].iat[-1] - df_steigung['hard_commitment_rate'].iat[-2]
    if arrow_value < 0:
        arrow_direction = "down"
    elif arrow_value > 0:
        arrow_direction = "up"
    else:
        arrow_direction = None
    
    return arrow_direction, arrow_value

# Liniengraf für Player ewm
def plot_player_ewm(df, name, df_trends, trend_events_count):
    df_player = df[df["user_name"] == name]
    
    fig = px.line(
        df_player,
        x="event_date_start",
        y=["soft_commitment_ewm", "hard_commitment_ewm"],
        labels={"value": "Commitment", "event_date_start": "Datum"},
        title=f"EWM Verlauf für {name}"
    )
    
    # Raw-Scores als Scatter hinzufügen
    fig.add_scatter(
        x=df_player["event_date_start"],
        y=df_player["soft_score"],
        mode="markers",
        name="soft raw"
    )
    fig.add_scatter(
        x=df_player["event_date_start"],
        y=df_player["hard_score"],
        mode="markers",
        name="hard raw"
    )
    
    # Trend als Graph einblenden
    if name in df_trends.index:
        trend_info = df_trends.loc[name]
        
        if trend_info["hard_trend"] in ["increasing", "decreasing"]:
            # Nur letzte N Trainings berücksichtigen
            last_n_dates = df_player["event_date_start"].unique()[-trend_events_count:]
            df_last_n = df_player[df_player["event_date_start"].isin(last_n_dates)]
            
            if len(df_last_n) >= 3:
                x_vals = df_last_n["event_date_start"]
                y_hard_trend = trend_info["intercept"] + trend_info["slope"] * np.arange(len(x_vals))
                
                fig.add_scatter(
                    x=x_vals,
                    y=y_hard_trend,
                    mode="lines",
                    line=dict(dash="dot", color="red"),
                    name="Hard Trend"
                )
    
    # Sichtbarkeit (Übersichtlicher dadurch)
    fig.update_traces(visible="legendonly", selector=dict(name="soft_commitment_ewm"))
    fig.update_traces(visible="legendonly", selector=dict(name="soft raw"))
    fig.update_traces(visible="legendonly", selector=dict(name="hard raw"))
    
    return fig

def large_kpi(value, label, arrow=None, arrow_value=None):
    arrow_symbol = ""
    if arrow == "up":
        arrow_symbol = "▲"
    elif arrow == "down":
        arrow_symbol = "▼"
    
    html = f"""
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        .kpi-large {{
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
            align-items: flex-start;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial;
            width: 100%;
            box-sizing: border-box;
            padding: 8px 6px;
            margin-top: 30px;
            color: #FFFFFF;
        }}

        .value-wrapper {{
            display: flex;
            align-items: center;
            gap: 8px;
            width: 100%;
        }}
        .value {{
            font-size: clamp(32px, 12vw, 72px);
            font-weight: 700;
            line-height: 1;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            -webkit-text-size-adjust: 100%;
        }}
        .label {{
            font-size: clamp(12px, 2.5vw, 20px);
            margin-top: 6px;
            opacity: 0.95;
        }}
        .kpi-arrow.up {{
            color: #00A86B;
            font-size: clamp(14px, 4vw, 22px);
        }}
        .kpi-arrow.down {{
            color: #FF4500;
            font-size: clamp(14px, 4vw, 22px);
        }}
    </style>

    <div class="kpi-large">
        <div class="value-wrapper">
            <div class="value">{value}</div>
            {f"<div class='kpi-arrow {arrow}'>{arrow_symbol} {arrow_value}</div>" if arrow else ""}
        </div>
        <div class="label">{label}</div>
    </div>
    """
    
    components.html(html, height=160)

def box_plot(grundwerte_df, spieler=None, box_points_selection="keine"):
    if box_points_selection == "keine":
        box_points = False
    elif box_points_selection == "alle":
        box_points = "all"
    elif box_points_selection == "Außreißer":
        box_points = "outliers"
    
    merkmale_quote = ["commitment_hard", "reliability"]
    merkmale_abs = ["confirmed"]
    merkmale = merkmale_quote + merkmale_abs
    
    df_long = grundwerte_df.reset_index().melt(
        id_vars="user_name",
        value_vars=merkmale,
        var_name="Merkmal",
        value_name="Wert"
    )
    
    # make_subplots mit sekundärer Y-Achse
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Boxplots für Quoten (links)
    for m in merkmale_quote:
        x_vals = df_long.loc[df_long["Merkmal"] == m, "Merkmal"]
        y_vals = df_long.loc[df_long["Merkmal"] == m, "Wert"]
        fig.add_trace(
            go.Box(
                x=x_vals,
                y=y_vals,
                name=m,
                boxpoints=box_points,
                jitter=0.4,
                pointpos=0,
                width=0.25,
                offsetgroup=m,
                alignmentgroup=m,
                marker_color="lightblue",
                text=df_long.loc[df_long["Merkmal"] == m, "user_name"],
                hoverinfo="text+y",
                boxmean=True,
            ),
            secondary_y=False
        )
    
    # Boxplots für absolute Werte (rechts)
    for m in merkmale_abs:
        x_vals = df_long.loc[df_long["Merkmal"] == m, "Merkmal"]
        y_vals = df_long.loc[df_long["Merkmal"] == m, "Wert"]
        fig.add_trace(
            go.Box(
                x=x_vals,
                y=y_vals,
                name=m,
                boxpoints=box_points,
                jitter=0.4,
                pointpos=0,
                width=0.25,
                offsetgroup=m,
                alignmentgroup=m,
                marker_color="lightgreen",
                text=df_long.loc[df_long["Merkmal"] == m, "user_name"],
                hoverinfo="text+y",
                boxmean=True,
            ),
            secondary_y=True
        )
    
    # Spieler hervorheben (nur wenn spieler angegeben)
    if spieler is not None:
        spieler_df = df_long[df_long["user_name"] == spieler]
        
        # Spieler hervorheben (Quoten)
        q = spieler_df[spieler_df["Merkmal"].isin(merkmale_quote)]
        if len(q) > 0:
            fig.add_trace(
                go.Scatter(
                    x=q["Merkmal"],
                    y=q["Wert"],
                    mode="markers+text",
                    marker=dict(size=14, color="red", symbol="diamond"),
                    text=[spieler] * len(q),
                    textposition="top center",
                    name=f"{spieler}",
                    offsetgroup="player_highlight",
                    alignmentgroup="player_highlight"
                ),
                secondary_y=False
            )
        
        # Spieler hervorheben (Absolute Werte)
        a = spieler_df[spieler_df["Merkmal"].isin(merkmale_abs)]
        if len(a) > 0:
            fig.add_trace(
                go.Scatter(
                    x=a["Merkmal"],
                    y=a["Wert"],
                    mode="markers+text",
                    marker=dict(size=14, color="red", symbol="diamond"),
                    text=[spieler] * len(a),
                    textposition="top center",
                    name=f"{spieler}",
                    offsetgroup="player_highlight",
                    alignmentgroup="player_highlight",
                    showlegend=False
                ),
                secondary_y=True
            )
    
    # Layout & Achsen
    title_text = "Boxplot der Merkmale (mit hervorgehobenem Spieler)" if spieler else "Boxplot der Merkmale"
    fig.update_layout(title=title_text, boxmode="group")
    
    # explizite Reihenfolge der Kategorien
    fig.update_xaxes(categoryorder="array", categoryarray=merkmale)
    
    fig.update_yaxes(title_text="Quote (0–1)", range=[0, 1], secondary_y=False)
    fig.update_yaxes(title_text="Absolute Werte", range=[0, max(grundwerte_df['confirmed']) + 5], secondary_y=True)
    
    if spieler:
        fig.update_layout(showlegend=False)
    
    return fig

def bar_plot(df_complete, players, merkmal, auto_zoom=False):
    # Daten filtern
    df_plot = df_complete.loc[players, [merkmal]].copy()
    df_plot = df_plot.reset_index()
    df_plot["Spieler"] = df_plot.iloc[:, 0]
    
    # Dynamische Y-Achse
    if auto_zoom:
        min_y = df_plot[merkmal].min() * 0.95
        max_y = df_plot[merkmal].max() * 1.05
        range_y = [min_y, max_y]
    else:
        range_y = [0, df_plot[merkmal].max() * 1.05]
    
    # Plot erstellen
    fig = px.bar(
        df_plot,
        x="Spieler",
        y=merkmal,
        title=f"{merkmal} Vergleich der Spieler",
        labels={merkmal: merkmal, "Spieler": "Spieler"},
    )
    
    # Y-Achse anpassen
    fig.update_yaxes(range=range_y)
    
    return fig

# Streamlit App
st.set_page_config(page_title="Trainingsbeteiligung Dashboard", layout="wide")

# Caching für Daten-Upload
@st.cache_data
def load_csv(uploaded_file):
    """Cache the uploaded CSV file"""
    return pd.read_csv(uploaded_file, sep=';')

@st.cache_data
def process_data(raw_data):
    """Cache the data processing"""
    if not check_search_params(raw_data):
        return None
    return filter_data(raw_data)

with st.sidebar:
    st.header("Daten hochladen")
    uploaded_file = st.file_uploader("Lade die CSV-Datei hoch", type=["csv"])
    
    with st.expander("Settings"):
        st.markdown("### 📅 Analysezeitraum")
        
        quick_range = st.selectbox(
            "Schnellauswahl",
            ["Komplett", "Letzte 30 Tage", "Letzte 90 Tage", "Dieses Jahr"],
            index=0
        )
        
        date_range = st.date_input("Oder manuell wählen", value=None)
        
        # Checkbox für erweiterte Daten
        show_tab4 = st.checkbox("Erweiterte Daten anzeigen", value=False)
        
        halflife_days = st.slider(
            "Halbwertszeit für EWM (Tage)",
            min_value=14,
            max_value=21,
            value=14,
            step=1
        )
        halflife_str = f"{halflife_days}d"
        
        trend_events_count = st.slider(
            "Trainingsanzahl zur Trendanalyse",
            min_value=8,
            max_value=20,
            value=10,
            step=1
        )
        
        time_team_commitment = st.selectbox(
            "Zeitframe Team Commitment",
            ["Tage", "Wochen", "Monate"],
            index=1,
        )
        
        median_team_commitment = st.checkbox("Median Team Commitment", value=False)
        
        box_points_selection = st.selectbox(
            "Punkte die im Boxplot angezeigt werden",
            ["keine", "alle", "Außreißer"],
            index=0,
        )

# Aufforderung daten hochladen
info_box = st.info("Daten hochladen um Dashboard anzuzeigen")

# Screen Größe sehen
screenD = ScreenData(setTimeout=1000)
screen_d = screenD.st_screen_data()

if uploaded_file is not None:
    info_box.empty()
    
    # Daten laden mit Caching
    raw_data = load_csv(uploaded_file)
    
    if not check_search_params(raw_data):
        st.error("Die Suchparameter enthalten nicht nur Trainingseinheiten. Bitte überprüfen Sie die Filtereinstellungen.")
    else:
        # Daten verarbeiten mit Caching
        data = process_data(raw_data)
        
        if data is not None:
            min_date = data['event_date_start'].min().date()
            max_date = data['event_date_start'].max().date()
            today = datetime.today().date()
            
            # Zeitraum festlegen
            if quick_range == "Letzte 30 Tage":
                start_date = max(min_date, today - timedelta(days=30))
                end_date = max_date
            elif quick_range == "Letzte 90 Tage":
                start_date = max(min_date, today - timedelta(days=90))
                end_date = max_date
            elif quick_range == "Dieses Jahr":
                start_date = max(min_date, datetime(today.year, 1, 1).date())
                end_date = max_date
            elif date_range and len(date_range) == 2:
                start_date, end_date = date_range
            else:
                start_date, end_date = min_date, max_date
            
            # GLOBAL FILTERN
            data_filtered = data[
                (data['event_date_start'].dt.date >= start_date) &
                (data['event_date_start'].dt.date <= end_date)
            ]
            
            # Grundwerte berechnen
            grundwerte_df = grundwerte(data_filtered)
            
            # Verlässlichkeit berechnen
            grundwerte_df['reliability'] = calculate_reliability(grundwerte_df['confirmed'], grundwerte_df['eligible_sessions'])
            
            # Commitment berechnen
            grundwerte_df['commitment_hard'], grundwerte_df['commitment_soft'] = commitment(grundwerte_df)
            
            # ewm Commitment berechnen
            df_prepared_ewm = prepare_data_ewm(data_filtered, player_start_dates)
            ewm_full = get_ewm_full(df_prepared_ewm, halflife=halflife_str)
            
            # Ewm Trend sehen
            all_player_trends = [detect_trend(ewm_full, player, trend_events_count, start_date, end_date) 
                               for player in ewm_full['user_name'].unique()]
            filtered_player_trends = [trend for trend in all_player_trends if trend is not None]
            
            # Create a DataFrame from the results
            df_trends = pd.DataFrame(filtered_player_trends).set_index('player_name') if filtered_player_trends else pd.DataFrame()
            
            # Dataframe mit allen Sachen
            df_complete = final_df(df_trends, ewm_full, grundwerte_df)
            
            if screen_d.get("innerWidth", 0) > 600:
                st.title("Trainingsbeteiligung Dashboard")
                
                # Basis-Tabs
                tab_labels = ['Teamanalyse', 'Individuelle Analyse', 'Spielervergleich', 'Teamvergleich']
                
                # Optional Tab 5 hinzufügen
                if show_tab4:
                    tab_labels.append('Erweiterte Daten')
                
                # Tabs erstellen
                tabs = st.tabs(tab_labels)
                
                # Zugriff auf einzelne Tabs
                tab1 = tabs[0]
                tab2 = tabs[1]
                tab3 = tabs[2]
                tab4 = tabs[3]
                
                if show_tab4:
                    tab5 = tabs[4]
                
                with tab1:
                    team_grundwerte_df = team_grundwerte(data_filtered)
                    
                    decreasing_trends_count = len(df_trends[df_trends['hard_trend'] == 'decreasing']) if not df_trends.empty else 0
                    
                    # --- Layout --- für KPIs
                    col_large, col_small = st.columns([2, 3])
                    
                    if not median_team_commitment:
                        large_kpi_value = int(round(team_grundwerte_df['hard_commitment_rate'].mean()))
                        large_kpi_label = "⌀ Hard Commitment / Training"
                    else:
                        large_kpi_value = int(round(team_grundwerte_df['hard_commitment_rate'].median()))
                        large_kpi_label = "Median Hard Commitment / Training"
                    
                    arrow_direction, arrow_value = arrow(team_grundwerte_df, time_team_commitment)
                    
                    # Großer KPI
                    with col_large:
                        if arrow_direction and arrow_value:
                            large_kpi(f"{large_kpi_value}%", large_kpi_label, arrow=arrow_direction, arrow_value=f"{round(arrow_value)}%")
                        else:
                            large_kpi(f"{large_kpi_value}%", large_kpi_label)
                    
                    # Kleine KPIs als Quadrate 2x2
                    small_kpis = [
                        {"label": "⌀ Spieleranzahl / Training", "value": round(avg_players_per_training(data_filtered))},
                        {"label": "Declining Players", "value": decreasing_trends_count},
                        {"label": "⌀ Hard Commitment / Spieler", "value": f"{round(grundwerte_df['commitment_hard'].mean()*100)}%"},
                        {"label": "Standardabweichung", "value": round(std_team(team_grundwerte_df), 2)}
                    ]
                    
                    with col_small:
                        row1_col1, row1_col2 = st.columns(2)
                        row2_col1, row2_col2 = st.columns(2)
                        
                        with row1_col1:
                            st.metric(small_kpis[0]["label"], small_kpis[0]["value"])
                        with row1_col2:
                            st.metric(small_kpis[1]["label"], small_kpis[1]["value"])
                        
                        with row2_col1:
                            st.metric(small_kpis[2]["label"], small_kpis[2]["value"])
                        with row2_col2:
                            st.metric(small_kpis[3]["label"], small_kpis[3]["value"])
                    
                    # Line Graph fürs team commitment je Trainingseinheit
                    st.plotly_chart(team_commitment_chart(team_grundwerte_df, time_team_commitment), use_container_width=True)
                
                with tab2:
                    player = st.selectbox("Spieler wählen", sorted(grundwerte_df.index.tolist()))
                    
                    col1, col2, col3 = st.columns(3)
                    
                    if player is not None and player in grundwerte_df.index:
                        com_h = float(grundwerte_df.loc[player, "commitment_hard"])
                        com_h = round(com_h * 100)
                        col1.metric("Hard Commitment", f"{com_h}%")
                        
                        # Verlässlichkeit zeigen
                        rel = float(grundwerte_df.loc[player, "reliability"])
                        rl = round(rel * 100)
                        col2.metric("Verlässlichkeit", f"{rl}%")
                    
                    # Plot für ewm und Trend
                    st.plotly_chart(plot_player_ewm(ewm_full, player, df_trends, trend_events_count), use_container_width=True)
                    
                    # Plot mit Box Charts
                    st.plotly_chart(box_plot(grundwerte_df, player, box_points_selection), use_container_width=True)
                    
                    st.expander("Advanced Stats").dataframe(grundwerte_df.loc[[player]])
                
                with tab3:
                    # 1️⃣ Spieler auswählen (Multiselect)
                    selected_players = st.multiselect(
                        "Mehrere Spieler auswählen",
                        grundwerte_df.index.tolist()
                    )
                    
                    # 2️⃣ Ausgewählte Spieler sortierbar machen
                    if selected_players:
                        with st.expander("Sortieren"):
                            selected_players = sort_items(
                                items=selected_players,
                                direction="horizontal",
                            )
                    
                    # 3️⃣ Merkmal auswählen
                    exclude_columns = ["total_sessions", "unsure", "not_choosed", "eligible_sessions", "intercept"]
                    merkmal_options = [col for col in df_complete.columns if col not in exclude_columns]
                    merkmal = st.selectbox("Merkmal auswählen", merkmal_options)
                    
                    # 4️⃣ Toggle für Y-Achse Auto-Zoom
                    auto_zoom = st.checkbox("Unterschiede sichtbar machen (bei nahen Werten)", value=False)
                    
                    # 5️⃣ Plot anzeigen
                    if selected_players and merkmal:
                        st.plotly_chart(bar_plot(df_complete, selected_players, merkmal, auto_zoom), use_container_width=True)
                
                with tab4:
                    st.header("Teamvergleich über verschiedene Zeiträume")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Zeitraum 1")
                        zeitraum1_typ = st.selectbox("Zeitraum 1 Auswahl", 
                                                    ["Letzte 30 Tage", "Letzte 90 Tage", "Dieses Jahr", "Manuell"],
                                                    key="zeitraum1_typ")
                        
                        if zeitraum1_typ == "Manuell":
                            zeitraum1 = st.date_input("Zeitraum 1 (manuell)", value=(min_date, max_date), key="zeitraum1")
                            if len(zeitraum1) == 2:
                                z1_start, z1_end = zeitraum1
                            else:
                                z1_start, z1_end = min_date, max_date
                        else:
                            if zeitraum1_typ == "Letzte 30 Tage":
                                z1_start = max(min_date, today - timedelta(days=30))
                                z1_end = max_date
                            elif zeitraum1_typ == "Letzte 90 Tage":
                                z1_start = max(min_date, today - timedelta(days=90))
                                z1_end = max_date
                            elif zeitraum1_typ == "Dieses Jahr":
                                z1_start = max(min_date, datetime(today.year, 1, 1).date())
                                z1_end = max_date
                    
                    with col2:
                        st.subheader("Zeitraum 2")
                        zeitraum2_typ = st.selectbox("Zeitraum 2 Auswahl",
                                                    ["Letzte 30 Tage", "Letzte 90 Tage", "Dieses Jahr", "Manuell"],
                                                    key="zeitraum2_typ")
                        
                        if zeitraum2_typ == "Manuell":
                            zeitraum2 = st.date_input("Zeitraum 2 (manuell)", value=(min_date, max_date), key="zeitraum2")
                            if len(zeitraum2) == 2:
                                z2_start, z2_end = zeitraum2
                            else:
                                z2_start, z2_end = min_date, max_date
                        else:
                            if zeitraum2_typ == "Letzte 30 Tage":
                                z2_start = max(min_date, today - timedelta(days=30))
                                z2_end = max_date
                            elif zeitraum2_typ == "Letzte 90 Tage":
                                z2_start = max(min_date, today - timedelta(days=90))
                                z2_end = max_date
                            elif zeitraum2_typ == "Dieses Jahr":
                                z2_start = max(min_date, datetime(today.year, 1, 1).date())
                                z2_end = max_date
                    
                    # Daten für beide Zeiträume filtern
                    data_z1 = data[(data['event_date_start'].dt.date >= z1_start) & 
                                  (data['event_date_start'].dt.date <= z1_end)]
                    data_z2 = data[(data['event_date_start'].dt.date >= z2_start) & 
                                  (data['event_date_start'].dt.date <= z2_end)]
                    
                    # Grundwerte für beide Zeiträume
                    grundwerte_z1 = grundwerte(data_z1)
                    grundwerte_z1['reliability'] = calculate_reliability(grundwerte_z1['confirmed'], grundwerte_z1['eligible_sessions'])
                    grundwerte_z1['commitment_hard'], grundwerte_z1['commitment_soft'] = commitment(grundwerte_z1)
                    
                    grundwerte_z2 = grundwerte(data_z2)
                    grundwerte_z2['reliability'] = calculate_reliability(grundwerte_z2['confirmed'], grundwerte_z2['eligible_sessions'])
                    grundwerte_z2['commitment_hard'], grundwerte_z2['commitment_soft'] = commitment(grundwerte_z2)
                    
                    # Boxplots nebeneinander
                    col_box1, col_box2 = st.columns(2)
                    
                    with col_box1:
                        st.plotly_chart(box_plot(grundwerte_z1, spieler=None, box_points_selection=box_points_selection), 
                                      use_container_width=True)
                        st.caption(f"Zeitraum 1: {z1_start} bis {z1_end}")
                    
                    with col_box2:
                        st.plotly_chart(box_plot(grundwerte_z2, spieler=None, box_points_selection=box_points_selection), 
                                      use_container_width=True)
                        st.caption(f"Zeitraum 2: {z2_start} bis {z2_end}")
                
                if show_tab4:
                    with tab5:
                        st.header("Rohdaten")
                        st.dataframe(raw_data)
                        
                        st.header("Gefilterte Daten")
                        st.dataframe(data_filtered)
                        
                        st.header("Alle erhobenen Werte")
                        st.dataframe(df_complete)
                        st.dataframe(df_complete.agg(["count", "mean", "median", "std", lambda x: x.std()/x.mean() if x.mean() != 0 else 0, "min", "max"]))
                        
                        st.header("EWM Trends")
                        st.dataframe(df_trends)
            
            # Dashboard auf dem Handy
            elif screen_d.get("innerWidth", 0) < 600:
                st.header("Trainingsbeteiligung")
                st.info("Für die vollständige Ansicht bitte einen größeren Bildschirm verwenden.")
                
                # Vereinfachte mobile Ansicht
                team_grundwerte_df = team_grundwerte(data_filtered)
                avg_commitment = int(round(team_grundwerte_df['hard_commitment_rate'].mean()))
                st.metric("⌀ Hard Commitment", f"{avg_commitment}%")
                
                player = st.selectbox("Spieler wählen", sorted(grundwerte_df.index.tolist()))
                if player in grundwerte_df.index:
                    com_h = int(round(float(grundwerte_df.loc[player, "commitment_hard"]) * 100))
                    st.metric(f"{player} - Hard Commitment", f"{com_h}%")
