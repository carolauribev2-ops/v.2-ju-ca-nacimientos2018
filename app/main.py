"""
Streamlit interface — Prediccion de Riesgo Perinatal.
Estilo visual basado en la identidad institucional del DANE Colombia.

Run:
    uv run streamlit run app/main.py
"""

from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="DANE · Riesgo Perinatal",
    page_icon="🇨🇴",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODELS_DIR = Path(__file__).parent.parent / "models"

# ── DANE color palette ─────────────────────────────────────────────────────────
# Extraído de https://www.dane.gov.co/ (logo-DANE-v_color.svg)
DANE_MAGENTA  = "#C0134E"   # color primario DANE
DANE_NAVY     = "#233979"   # azul oscuro (bandera Colombia)
DANE_GOLD     = "#FEC800"   # dorado (bandera Colombia)
DANE_RED      = "#DB0D1F"   # rojo (bandera Colombia)
DANE_LIGHT_BG = "#F5F6FA"   # fondo claro
DANE_CARD_BG  = "#FFFFFF"
DANE_BORDER   = "#E0E4EE"
DANE_TEXT     = "#1A1A2E"
DANE_MUTED    = "#6B7280"

# ── Reference data ─────────────────────────────────────────────────────────────

DEPARTAMENTOS: dict[str, int] = {
    "Bogota D.C.": 11,
    "Antioquia": 5,
    "Valle del Cauca": 76,
    "Atlantico": 8,
    "Bolivar": 13,
    "Cundinamarca": 25,
    "Santander": 68,
    "Cordoba": 23,
    "Narino": 52,
    "Norte de Santander": 54,
    "Cauca": 19,
    "Boyaca": 15,
    "Tolima": 73,
    "Huila": 41,
    "Magdalena": 47,
    "Cesar": 20,
    "Meta": 50,
    "La Guajira": 44,
    "Sucre": 70,
    "Caldas": 17,
    "Risaralda": 66,
    "Quindio": 63,
    "Choco": 27,
    "Casanare": 85,
    "Caqueta": 18,
    "Putumayo": 86,
    "Arauca": 81,
    "Amazonas": 91,
    "Guainia": 94,
    "Guaviare": 95,
    "Vaupes": 97,
    "Vichada": 99,
    "San Andres y Providencia": 88,
    "Desconocido / Exterior": -1,
}

AREA_OPTIONS: dict[str, int] = {
    "Cabecera Municipal (Urbana)": 1,
    "Centro Poblado": 2,
    "Rural Disperso": 3,
    "Desconocido": -1,
}

REGIMEN_OPTIONS: dict[str, int] = {
    "Contributivo (EPS privada)": 1,
    "Subsidiado (SISBEN)": 2,
    "Especial (FF.MM / Policia)": 3,
    "Excepcion (Ecopetrol / Magisterio)": 4,
    "No asegurado / Sin afiliacion": 5,
    "Desconocido": -1,
}

MUL_PARTO_OPTIONS: dict[str, int] = {
    "Unico (simple)": 1,
    "Gemelar (mellizos)": 2,
    "Multiple (trillizos o mas)": 3,
}

# DANE EDAD_MADRE categorical codes
EDAD_MADRE_OPTIONS: dict[str, int] = {
    "Menor de 15 anos": 1,
    "15 - 19 anos": 2,
    "20 - 24 anos": 3,
    "25 - 29 anos": 4,
    "30 - 34 anos": 5,
    "35 - 39 anos": 6,
    "40 - 44 anos": 7,
    "45 - 49 anos": 8,
    "50 anos o mas": 9,
}

# DANE T_GES categorical codes
T_GES_OPTIONS: dict[str, int] = {
    "22 - 27 semanas (muy prematuro)": 2,
    "28 - 36 semanas (prematuro)": 3,
    "37 - 41 semanas (termino)": 4,
    "42 o mas semanas (postermino)": 5,
    "Sin informacion": 9,
}

# ── CSS & Branding ─────────────────────────────────────────────────────────────

DANE_CSS = f"""
<style>
  /* ── Typography & base ── */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

  html, body, [class*="css"] {{
    font-family: 'Inter', 'Segoe UI', Arial, sans-serif;
    color: {DANE_TEXT};
  }}

  /* ── Hide default Streamlit chrome ── */
  #MainMenu, footer {{ visibility: hidden; }}
  header {{ background: transparent !important; }}
  [data-testid="stHeaderActionElements"] {{ display: none; }}
  .block-container {{ padding-top: 0 !important; }}

  /* ── App background ── */
  .stApp {{
    background-color: {DANE_LIGHT_BG};
  }}

  /* ── Custom header bar ── */
  .dane-topbar {{
    background: #FFFFFF;
    border-bottom: 4px solid {DANE_MAGENTA};
    padding: 22px 32px;
    margin: -1px -1px 28px -1px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 28px;
    box-shadow: 0 2px 10px rgba(35,57,121,0.10);
  }}
  .dane-brand {{
    display: flex;
    align-items: center;
    gap: 22px;
    min-width: 0;
  }}
  .dane-logo {{
    display: flex;
    flex-direction: column;
    align-items: center;
    flex: 0 0 auto;
  }}
  .dane-logo-word {{
    color: {DANE_NAVY};
    font-size: 40px;
    font-weight: 850;
    letter-spacing: -2px;
    line-height: 1;
  }}
  .dane-logo-accent {{
    width: 82%;
    height: 4px;
    margin-top: 5px;
    border-radius: 3px;
    background: {DANE_MAGENTA};
  }}
  .dane-brand-copy {{
    border-left: 1px solid {DANE_BORDER};
    padding-left: 22px;
  }}
  .dane-brand-eyebrow {{
    margin: 0 0 5px;
    color: {DANE_MAGENTA};
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
  }}
  .dane-brand-copy h1 {{
    margin: 0;
    color: {DANE_NAVY};
    font-size: clamp(22px, 3vw, 32px);
    font-weight: 750;
    letter-spacing: -0.035em;
    line-height: 1.15;
  }}
  .dane-brand-subtitle {{
    margin: 6px 0 0;
    color: {DANE_MUTED};
    font-size: 13px;
  }}
  .dane-topbar-right {{
    text-align: right;
    color: {DANE_MUTED};
    font-size: 12px;
    line-height: 1.6;
    flex: 0 0 auto;
  }}
  .dane-topbar-right strong {{
    color: {DANE_MAGENTA};
    font-size: 10px;
    font-weight: 700;
    display: block;
    letter-spacing: 0.12em;
    text-transform: uppercase;
  }}
  .dane-topbar-right span {{
    display: block;
    color: {DANE_NAVY};
    font-size: 13px;
    font-weight: 600;
  }}
  @media (max-width: 720px) {{
    .dane-topbar {{
      align-items: flex-start;
      flex-direction: column;
      gap: 18px;
      padding: 20px;
    }}
    .dane-brand {{
      align-items: flex-start;
      gap: 16px;
    }}
    .dane-logo-word {{
      font-size: 32px;
    }}
    .dane-brand-copy {{
      padding-left: 16px;
    }}
    .dane-topbar-right {{
      padding-left: 16px;
      text-align: left;
      border-left: 2px solid {DANE_BORDER};
    }}
  }}

  /* ── Flag ribbon ── */
  .dane-flag-ribbon {{
    height: 8px;
    background: linear-gradient(to right,
      {DANE_GOLD}   0%   33.3%,
      {DANE_RED}    33.3% 66.6%,
      {DANE_NAVY}   66.6% 100%
    );
    margin: 0;
  }}

  /* ── Section headings ── */
  .dane-section-title {{
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 16px;
  }}
  .dane-section-bar {{
    width: 5px;
    height: 28px;
    background: {DANE_MAGENTA};
    border-radius: 3px;
    flex-shrink: 0;
  }}
  .dane-section-text {{
    font-size: 17px;
    font-weight: 700;
    color: {DANE_NAVY};
    margin: 0;
  }}

  /* ── Prediction cards ── */
  .dane-card {{
    background: {DANE_CARD_BG};
    border-radius: 10px;
    border: 1px solid {DANE_BORDER};
    border-top: 4px solid {DANE_MAGENTA};
    padding: 20px 22px 16px;
    box-shadow: 0 2px 10px rgba(35,57,121,0.08);
    margin-bottom: 16px;
  }}
  .dane-card-navy {{
    border-top-color: {DANE_NAVY};
  }}

  /* ── Risk badge ── */
  .risk-badge {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 16px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.5px;
    margin-top: 4px;
  }}
  .risk-low      {{ background: #D4F4DD; color: #155724; border: 1px solid #A8DDB8; }}
  .risk-moderate {{ background: #FFF3CD; color: #7B4F00; border: 1px solid #FFD77A; }}
  .risk-high     {{ background: #FBDDE2; color: #7B0015; border: 1px solid #F5B0BA; }}

  /* ── Probability display ── */
  .prob-number {{
    font-size: 48px;
    font-weight: 800;
    line-height: 1;
    margin: 8px 0 4px;
  }}
  .prob-low      {{ color: #1B7A38; }}
  .prob-moderate {{ color: #B86A00; }}
  .prob-high     {{ color: {DANE_MAGENTA}; }}

  /* ── Factor alert item ── */
  .factor-item {{
    display: flex;
    align-items: flex-start;
    gap: 8px;
    background: #FFF8F0;
    border-left: 3px solid {DANE_GOLD};
    border-radius: 4px;
    padding: 8px 12px;
    margin-bottom: 6px;
    font-size: 13px;
    color: {DANE_TEXT};
  }}

  /* ── Summary table ── */
  .dane-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid {DANE_BORDER};
  }}
  .dane-table th {{
    background: {DANE_NAVY};
    color: white;
    padding: 10px 14px;
    text-align: left;
    font-weight: 600;
    font-size: 12px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
  }}
  .dane-table td {{
    padding: 9px 14px;
    border-bottom: 1px solid {DANE_BORDER};
    vertical-align: top;
  }}
  .dane-table tr:nth-child(even) td {{ background: #F9FAFF; }}
  .dane-table tr:hover td {{ background: #EFF2FF; transition: background 0.15s; }}
  .dane-table td:first-child {{ font-weight: 600; color: {DANE_NAVY}; }}
  .dane-table td:last-child {{ color: {DANE_MUTED}; font-size: 12px; }}

  /* ── Sidebar ── */
  section[data-testid="stSidebar"] {{
    background-color: {DANE_CARD_BG};
    border-right: 3px solid {DANE_MAGENTA};
  }}
  section[data-testid="stSidebar"] .stSelectbox label,
  section[data-testid="stSidebar"] .stSlider label,
  section[data-testid="stSidebar"] .stRadio label {{
    font-weight: 600;
    font-size: 13px;
    color: {DANE_NAVY};
  }}
  section[data-testid="stSidebar"] h1,
  section[data-testid="stSidebar"] h2,
  section[data-testid="stSidebar"] h3 {{
    color: {DANE_NAVY};
    border-bottom: 2px solid {DANE_MAGENTA};
    padding-bottom: 4px;
  }}

  /* ── Selectbox & slider accent ── */
  .stSelectbox [data-baseweb="select"] {{
    border-color: {DANE_BORDER} !important;
  }}
  .stSlider [data-baseweb="slider"] {{
    color: {DANE_MAGENTA} !important;
  }}

  /* ── Divider ── */
  .dane-divider {{
    height: 2px;
    background: linear-gradient(to right, {DANE_MAGENTA}, {DANE_NAVY}, transparent);
    border: none;
    margin: 24px 0;
    border-radius: 2px;
  }}

  /* ── Footer ── */
  .dane-footer {{
    background: {DANE_NAVY};
    color: rgba(255,255,255,0.7);
    padding: 16px 24px;
    border-radius: 8px;
    font-size: 12px;
    line-height: 1.7;
    margin-top: 24px;
  }}
  .dane-footer strong {{ color: {DANE_GOLD}; }}
  .dane-footer a {{ color: {DANE_GOLD}; text-decoration: none; }}

  /* ── No-model alert ── */
  .dane-alert {{
    background: #FFF3CD;
    border: 1px solid {DANE_GOLD};
    border-left: 5px solid {DANE_GOLD};
    border-radius: 6px;
    padding: 16px 20px;
    margin: 16px 0;
  }}
</style>
"""

# ── Model loading ──────────────────────────────────────────────────────────────


@st.cache_resource
def load_models() -> dict:
    loaded = {}
    for key, filename in [
        ("cesarea", "cesarea_pipeline.pkl"),
        ("bajo_peso", "bajo_peso_pipeline.pkl"),
    ]:
        path = MODELS_DIR / filename
        if path.exists():
            loaded[key] = joblib.load(path)
    return loaded


# ── UI helpers ─────────────────────────────────────────────────────────────────


def gauge_chart(prob: float, title: str, primary_color: str) -> go.Figure:
    pct = prob * 100
    if pct < 30:
        bar_color = "#1B7A38"
    elif pct < 60:
        bar_color = "#B86A00"
    else:
        bar_color = DANE_MAGENTA

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        title={
            "text": title,
            "font": {"size": 14, "color": DANE_NAVY, "family": "Inter, Arial"},
        },
        number={
            "suffix": "%",
            "font": {"size": 36, "color": bar_color, "family": "Inter, Arial"},
            "valueformat": ".1f",
        },
        gauge={
            "axis": {
                "range": [0, 100],
                "ticksuffix": "%",
                "tickfont": {"size": 10, "color": DANE_MUTED},
                "tickcolor": DANE_BORDER,
            },
            "bar": {"color": bar_color, "thickness": 0.28},
            "bgcolor": DANE_CARD_BG,
            "borderwidth": 1,
            "bordercolor": DANE_BORDER,
            "steps": [
                {"range": [0, 30],  "color": "#E8F7ED"},
                {"range": [30, 60], "color": "#FFF8E7"},
                {"range": [60, 100], "color": "#FDE8EE"},
            ],
            "threshold": {
                "line": {"color": DANE_MAGENTA, "width": 2},
                "thickness": 0.75,
                "value": 60,
            },
        },
    ))
    fig.update_layout(
        height=240,
        margin=dict(l=16, r=16, t=52, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, Arial"},
    )
    return fig


def risk_html(prob: float) -> str:
    if prob < 0.30:
        return '<span class="risk-badge risk-low">&#9679; RIESGO BAJO</span>'
    elif prob < 0.60:
        return '<span class="risk-badge risk-moderate">&#9679; RIESGO MODERADO</span>'
    return '<span class="risk-badge risk-high">&#9679; RIESGO ALTO</span>'


def prob_html(prob: float) -> str:
    pct = prob * 100
    cls = "prob-low" if pct < 30 else ("prob-moderate" if pct < 60 else "prob-high")
    return f'<div class="prob-number {cls}">{pct:.1f}<span style="font-size:24px">%</span></div>'


def factors_html(factors: list[str]) -> str:
    if not factors:
        return (
            '<div class="factor-item" style="border-left-color:#1B7A38;background:#E8F7ED;">'
            '&#10003; No se identificaron factores de riesgo adicionales.</div>'
        )
    return "".join(
        f'<div class="factor-item">&#9888; {f}</div>' for f in factors
    )


def cesarea_risk_factors(
    edad_code: int,
    t_ges_code: int,
    mul: int,
    n_emb: int,
    numconsul: int,
    area_res: int,
    idclasadmi: int,
    n_hijosv: int,
    prob: float,
) -> list[str]:
    factors = []

    # ── Edad materna ──
    if edad_code == 1:
        factors.append("Madre menor de 15 anos — desproporcion cefalopelvica frecuente en adolescentes")
    elif edad_code in {7, 8, 9}:
        factors.append("Edad materna avanzada (>= 40 anos) — mayor probabilidad de cesarea programada")
    elif edad_code == 6:
        factors.append("Edad materna 35-39 anos — riesgo obstetricio moderadamente elevado")

    # ── Gestacion ──
    if t_ges_code in {2, 3}:
        factors.append("Parto prematuro (< 37 semanas) — indicacion medica frecuente de cesarea")
    elif t_ges_code == 5:
        factors.append("Gestacion postermino (>= 42 semanas) — mayor riesgo de sufrimiento fetal")

    # ── Paridad y embarazos ──
    if mul > 1:
        factors.append("Gestacion multiple (gemelar/multiple) — indicacion habitual de cesarea")
    if n_emb > 3:
        factors.append("Alta paridad (> 3 embarazos previos) — mayor riesgo de complicaciones")
    if n_hijosv >= 3:
        factors.append("3 o mas hijos vivos — posible antecedente de cesarea iterativa")

    # ── Control prenatal ──
    if numconsul == 0:
        factors.append("Sin consultas prenatales — ausencia total de seguimiento obstetrico")
    elif numconsul < 4:
        factors.append(f"Control prenatal insuficiente ({numconsul} consultas) — minimo recomendado: 4")

    # ── Area y acceso ──
    if area_res == 2:
        factors.append("Residencia en centro poblado — mayor medicalizacion del parto en zonas semi-urbanas")
    elif area_res == 3:
        factors.append("Residencia rural — menor acceso a parto natural asistido y mayor derivacion hospitalaria")

    # ── Regimen de salud ──
    if idclasadmi in {5, -1}:
        factors.append("Sin cobertura en salud — seguimiento prenatal irregular o ausente")
    elif idclasadmi in {3, 4}:
        factors.append("Regimen especial (FF.MM./Ecopetrol/Magisterio) — mayor tasa de cesarea programada en estos regimenes")

    # ── Nota cuando la probabilidad es alta sin factores individuales criticos ──
    if not factors and prob >= 0.50:
        factors.append(
            f"La probabilidad ({prob:.0%}) supera la media nacional (~44.7%). "
            "La combinacion de caracteristicas demograficas, departamento y acceso al sistema "
            "de salud eleva el riesgo, aunque no se detectan factores clinicos individuales criticos."
        )

    return factors


def bajo_peso_risk_factors(
    edad_code: int,
    t_ges_code: int,
    mul: int,
    area: int,
    numconsul: int,
    regimen: int,
    n_emb: int,
    prob: float,
) -> list[str]:
    factors = []

    # ── Principal predictor: prematuridad ──
    if t_ges_code == 2:
        factors.append("Prematuridad extrema (22-27 semanas) — causa directa de bajo peso severo")
    elif t_ges_code == 3:
        factors.append("Prematuridad (28-36 semanas) — principal causa de bajo peso al nacer")

    # ── Edad materna ──
    if edad_code == 1:
        factors.append("Madre menor de 15 anos — mayor riesgo nutricional y de bajo peso neonatal")
    elif edad_code in {7, 8, 9}:
        factors.append("Edad materna avanzada (>= 40 anos) — mayor riesgo de restriccion del crecimiento fetal")

    # ── Gestacion multiple ──
    if mul == 2:
        factors.append("Gestacion gemelar — competencia de recursos placentarios entre fetos")
    elif mul >= 3:
        factors.append("Gestacion multiple (trillizos o mas) — alto riesgo de bajo peso por competencia fetal")

    # ── Paridad alta ──
    if n_emb >= 5:
        factors.append("Gran multiparidad (>= 5 embarazos) — menor reserva nutricional materna")

    # ── Control prenatal ──
    if numconsul == 0:
        factors.append("Sin consultas prenatales — ausencia total de seguimiento nutricional y clinico")
    elif numconsul < 4:
        factors.append(f"Control prenatal insuficiente ({numconsul} consultas) — deteccion tardia de restriccion del crecimiento")

    # ── Area de residencia ──
    if area == 3:
        factors.append("Residencia rural — menor acceso a atencion prenatal, nutricion y suplementacion")
    elif area == 2:
        factors.append("Centro poblado — acceso limitado a especialistas y seguimiento nutricional")

    # ── Cobertura en salud ──
    if regimen in {5, -1}:
        factors.append("Sin cobertura en salud — menor acceso a controles prenatales y suplementacion")

    # ── Nota cuando la probabilidad es alta sin factores criticos ──
    if not factors and prob >= 0.15:
        factors.append(
            f"La probabilidad ({prob:.0%}) supera la media nacional (~9.1%). "
            "La combinacion de caracteristicas del embarazo y acceso al sistema de salud "
            "eleva el riesgo sobre la media poblacional."
        )

    return factors


# ── Main app ───────────────────────────────────────────────────────────────────


def main() -> None:
    # Inject styles
    st.markdown(DANE_CSS, unsafe_allow_html=True)

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <header class="dane-topbar">
          <div class="dane-brand">
            <div class="dane-logo" role="img" aria-label="DANE">
              <span class="dane-logo-word" aria-hidden="true">DANE</span>
              <span class="dane-logo-accent" aria-hidden="true"></span>
            </div>
            <div class="dane-brand-copy">
              <p class="dane-brand-eyebrow">Estadísticas vitales · Colombia 2018</p>
              <h1>Predicción de riesgo perinatal</h1>
              <p class="dane-brand-subtitle">Cesárea · Bajo peso al nacer</p>
            </div>
          </div>
          <div class="dane-topbar-right">
            <strong>Herramienta de apoyo</strong>
            <span>Microdatos DANE</span>
          </div>
        </header>
        """,
        unsafe_allow_html=True,
    )

    models = load_models()

    if not models:
        st.markdown(
            """
            <div class="dane-alert">
              <strong>&#9888; Modelos no encontrados</strong><br>
              Ejecute primero el entrenamiento:<br>
              <code>uv run python main.py</code>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            f"""
            <div style="background:{DANE_MAGENTA};color:white;padding:10px 14px;
                        border-radius:6px;margin-bottom:16px;font-weight:700;font-size:14px;">
              &#128203; Datos del Embarazo
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("**Madre**")
        edad_label = st.selectbox(
            "Edad de la madre",
            list(EDAD_MADRE_OPTIONS.keys()),
            index=3,
        )
        edad_madre = EDAD_MADRE_OPTIONS[edad_label]

        area_label = st.selectbox("Area de residencia", list(AREA_OPTIONS.keys()))
        area_res = AREA_OPTIONS[area_label]
        regimen_label = st.selectbox("Regimen de salud", list(REGIMEN_OPTIONS.keys()))
        idclasadmi = REGIMEN_OPTIONS[regimen_label]
        depto_label = st.selectbox("Departamento de residencia", list(DEPARTAMENTOS.keys()))
        codptore = DEPARTAMENTOS[depto_label]

        st.markdown("---")
        st.markdown("**Embarazo**")
        t_ges_label = st.selectbox(
            "Semanas de gestacion",
            list(T_GES_OPTIONS.keys()),
            index=2,
        )
        t_ges = T_GES_OPTIONS[t_ges_label]
        numconsul = st.slider("Consultas prenatales", min_value=0, max_value=20, value=6)
        n_emb = st.slider("N.° de embarazos (incluye el actual)", min_value=1, max_value=20, value=1)
        n_hijosv = st.slider("Hijos vivos actualmente", min_value=1, max_value=15, value=1)
        mul_label = st.selectbox("Tipo de gestacion", list(MUL_PARTO_OPTIONS.keys()))
        mul_parto = MUL_PARTO_OPTIONS[mul_label]

        st.markdown("---")
        st.markdown("**Bebe**")
        sexo_label = st.radio("Sexo del bebe", ["Masculino", "Femenino"], horizontal=True)
        sexo = 1 if sexo_label == "Masculino" else 2

    # ── Prediction input ──────────────────────────────────────────────────────
    input_df = pd.DataFrame([{
        "EDAD_MADRE": edad_madre,
        "T_GES": t_ges,
        "NUMCONSUL": numconsul,
        "N_HIJOSV": n_hijosv,
        "N_EMB": n_emb,
        "AREA_RES": area_res,
        "IDCLASADMI": idclasadmi,
        "SEXO": sexo,
        "MUL_PARTO": mul_parto,
        "CODPTORE": codptore,
    }])

    # ── Results ───────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown(
            '<div class="dane-section-title">'
            '<div class="dane-section-bar"></div>'
            '<p class="dane-section-text">Probabilidad de Cesarea</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        if "cesarea" not in models:
            st.warning("Modelo no disponible.")
        else:
            prob_c = float(models["cesarea"].predict_proba(input_df)[0][1])

            st.markdown('<div class="dane-card">', unsafe_allow_html=True)
            st.plotly_chart(
                gauge_chart(prob_c, "Probabilidad de Cesarea", DANE_MAGENTA),
                use_container_width=True,
            )
            st.markdown(prob_html(prob_c), unsafe_allow_html=True)
            st.markdown(risk_html(prob_c), unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            with st.expander("Ver factores de riesgo"):
                factors = cesarea_risk_factors(
                    edad_madre, t_ges, mul_parto, n_emb, numconsul,
                    area_res, idclasadmi, n_hijosv, prob_c,
                )
                st.markdown(factors_html(factors), unsafe_allow_html=True)

    with col2:
        st.markdown(
            '<div class="dane-section-title">'
            '<div class="dane-section-bar" style="background:#233979"></div>'
            '<p class="dane-section-text">Riesgo de Bajo Peso al Nacer</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        if "bajo_peso" not in models:
            st.warning("Modelo no disponible.")
        else:
            prob_bp = float(models["bajo_peso"].predict_proba(input_df)[0][1])

            st.markdown('<div class="dane-card dane-card-navy">', unsafe_allow_html=True)
            st.plotly_chart(
                gauge_chart(prob_bp, "Riesgo de Bajo Peso", DANE_NAVY),
                use_container_width=True,
            )
            st.markdown(prob_html(prob_bp), unsafe_allow_html=True)
            st.markdown(risk_html(prob_bp), unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            with st.expander("Ver factores de riesgo"):
                factors = bajo_peso_risk_factors(
                    edad_madre, t_ges, mul_parto, area_res, numconsul,
                    idclasadmi, n_emb, prob_bp,
                )
                st.markdown(factors_html(factors), unsafe_allow_html=True)

    # ── Summary table ─────────────────────────────────────────────────────────
    st.markdown('<hr class="dane-divider"/>', unsafe_allow_html=True)
    st.markdown(
        '<div class="dane-section-title">'
        '<div class="dane-section-bar" style="background:#FEC800"></div>'
        '<p class="dane-section-text">Resumen de la Evaluacion</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    rows = [
        ("Edad materna",         edad_label,                "Riesgo si &lt;15 o &ge;40 anos"),
        ("Semanas de gestacion", t_ges_label,               "Normal: 37-41 semanas"),
        ("Consultas prenatales", f"{numconsul} consultas",  "Minimo recomendado: 4"),
        ("N.° embarazos",        str(n_emb),                ""),
        ("Hijos vivos",          str(n_hijosv),             ""),
        ("Tipo de gestacion",    mul_label,                 "Multiple = mayor riesgo"),
        ("Area de residencia",   area_label,                "Mayor acceso en zona urbana"),
        ("Regimen de salud",     regimen_label,             "Contributivo/Subsidiado: mayor cobertura"),
        ("Departamento",         depto_label,               ""),
    ]

    rows_html = "".join(
        f"<tr><td>{p}</td><td>{v}</td><td>{r}</td></tr>"
        for p, v, r in rows
    )
    st.markdown(
        f"""
        <table class="dane-table">
          <thead>
            <tr>
              <th>Parametro</th>
              <th>Valor ingresado</th>
              <th>Referencia clinica</th>
            </tr>
          </thead>
          <tbody>{rows_html}</tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div class="dane-footer">
          <strong>&#9888; Aviso de uso:</strong> Esta herramienta es de apoyo a la
          decision clinica y <strong>no reemplaza el criterio medico profesional</strong>.
          Las predicciones se basan en patrones estadisticos de los registros de
          nacimientos de Colombia en 2018.<br><br>
          <strong>Fuente de datos:</strong>
          <a href="https://microdatos.dane.gov.co/index.php/catalog/652" target="_blank">
            DANE — Estadisticas Vitales, Microdatos Nacimientos 2018
          </a>
          &nbsp;|&nbsp;
          <strong>Registros analizados:</strong> 648 154
          &nbsp;|&nbsp;
          <strong>Tasa de cesarea:</strong> 44.4 %
          &nbsp;|&nbsp;
          <strong>Tasa bajo peso:</strong> 9.1 %
          
          <hr style="border-color: rgba(255,255,255,0.15); margin: 14px 0;">
          
          <div style="display: flex; justify-content: space-between; align-items: center; font-size: 11.5px;">
            <span>Proyecto academico — Especializacion en Ciencias de Datos e Inteligencia Artificial</span>
            <span>Desarrollado por: <strong>Juanita Arango</strong> y <strong>Carolina Uribe</strong></span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
