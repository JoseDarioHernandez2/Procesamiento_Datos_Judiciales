import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path
import warnings
import re

warnings.filterwarnings("ignore")
st.set_page_config(page_title="DashBoard Inteligencia Penitenciaria v4.1", page_icon="🚨", layout="wide")

# ─────────────────────────────────────────────────────────────────────────────
# 1. ESTILOS
# ─────────────────────────────────────────────────────────────────────────────
def inject_styles():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@700;800;900&family=Barlow:wght@400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Barlow', sans-serif; font-size: 15px; }

    [data-testid="stSidebar"] { background: #0D2535; }
    [data-testid="stSidebar"] * { color: #CBD5E1 !important; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 { color: #fff !important; font-size: 1.2rem !important; }
    [data-testid="stSidebar"] label { font-size: 0.88rem !important; font-weight: 700 !important;
      text-transform: uppercase; letter-spacing: .04em; }

    .main-header { background: linear-gradient(135deg, #0D2535 0%, #1565C0 100%);
      color: white; padding: 1.3rem 2rem; border-radius: 14px; margin-bottom: 1.2rem;
      box-shadow: 0 6px 24px rgba(13,37,53,.35); }
    .main-header h1 { font-family:'Barlow Condensed',sans-serif; font-size: 2.2rem;
      font-weight: 900; margin: 0; text-transform: uppercase; letter-spacing: .05em; }
    .main-header p  { margin: .25rem 0 0; font-size: 1.05rem; opacity: .85; }

    .tot-box { background:#0D2535; color:white; border-radius:10px;
      padding:.75rem 1rem; text-align:center; }
    .tot-num { font-family:'Barlow Condensed',sans-serif; font-size:2.1rem;
      font-weight:900; color:#38BDF8; line-height:1; }
    .tot-lbl { font-size:.75rem; font-weight:700; text-transform:uppercase;
      letter-spacing:.06em; color:#94A3B8; margin-top:.2rem; }

    .kpi-card { background:white; border-radius:12px; padding:1.1rem 1.3rem;
      box-shadow:0 3px 14px rgba(0,0,0,.1); border-left:6px solid #1565C0; }
    .kpi-card.rojo    { border-left-color:#DC2626; background:#FFF1F2; }
    .kpi-card.verde   { border-left-color:#16A34A; background:#F0FDF4; }
    .kpi-card.naranja { border-left-color:#EA580C; background:#FFF7ED; }
    .kpi-lbl { font-size:.78rem; font-weight:700; text-transform:uppercase;
      letter-spacing:.06em; color:#64748B; }
    .kpi-val { font-family:'Barlow Condensed',sans-serif; font-size:2.2rem;
      font-weight:900; color:#0D2535; line-height:1.1; margin:.12rem 0; }
    .kpi-card.rojo .kpi-val    { color:#DC2626; }
    .kpi-card.verde .kpi-val   { color:#16A34A; }
    .kpi-card.naranja .kpi-val { color:#EA580C; }
    .kpi-sub { font-size:.82rem; color:#64748B; }

    .stTabs [data-baseweb="tab-list"] { gap:6px; }
    .stTabs [data-baseweb="tab"] { background:#E2E8F0; border-radius:8px 8px 0 0;
      padding:10px 18px; font-size:1rem; font-weight:700; color:#334155; }
    .stTabs [aria-selected="true"] { background:#1565C0 !important; color:white !important; }

    .sec { font-family:'Barlow Condensed',sans-serif; font-size:1.3rem; font-weight:800;
      color:#0D2535; text-transform:uppercase; letter-spacing:.05em;
      border-bottom:3px solid #1565C0; padding-bottom:.3rem; margin:1rem 0 .7rem; }

    .footer { background:#0D2535; color:#64748B; text-align:center;
      font-size:.8rem; padding:.55rem; border-radius:8px; margin-top:1.5rem; }

    /* CRÉDITO DORADO */
    .footer-container {
        text-align: center; padding-top: 20px;
        border-top: 1px solid rgba(255,255,255,0.1); margin-top: 30px;
    }
    .footer-title { font-size:13px; color:#CBD5E1; margin-bottom:5px;
      letter-spacing:0.5px; font-weight:600; }
    .footer-experts { font-size:11px; color:#94A3B8; font-style:italic; margin-bottom:12px; }
    .footer-names { font-size:14px; font-weight:800;
      background:linear-gradient(135deg,#D4AF37 0%,#F4D03F 100%);
      -webkit-background-clip:text; -webkit-text-fill-color:transparent;
      margin-bottom:15px; font-family:'Barlow Condensed',sans-serif; }
    .footer-uni { font-size:10px; color:#64748B; line-height:1.4;
      text-transform:uppercase; font-weight:700; }
    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 2. CARGA DE DATOS — blindaje numérico con np.nan (nunca pd.NA)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    base = Path("DWH")
    fact       = pd.read_csv(base / "fact_detenidos.csv")
    dim_rep    = pd.read_csv(base / "dim_reporte.csv")
    dim_ubi    = pd.read_csv(base / "dim_ubicacion.csv")
    dim_tiempo = pd.read_csv(base / "dim_tiempo.csv")

    for d in [fact, dim_rep, dim_ubi, dim_tiempo]:
        d.columns = d.columns.str.strip().str.lower()

    df = (fact
          .merge(dim_rep,    on="id_reporte",   how="left")
          .merge(dim_ubi,    on="id_ubicacion",  how="left")
          .merge(dim_tiempo, on="id_tiempo",     how="left"))

    # Responsable = rango + operador
    rango = df["operador_rango"].fillna("") if "operador_rango" in df.columns else ""
    oper  = df["operador"].fillna("")        if "operador"       in df.columns else ""
    df["responsable"] = (rango + " " + oper).str.strip()

    # Mapeo regionales
    rmap = {
        "CENTRO":"Regional 1 - Cundinamarca",   "SUR":"Regional 2 - Huila/Tolima",
        "EJE CAFETERO":"Regional 3 - Risaralda","OCCIDENTE":"Regional 4 - Valle",
        "ORIENTE":"Regional 5 - Santander",      "ANTIOQUIA":"Regional 6 - Antioquia",
        "LLANOS":"Regional 7 - Meta/Llanos",     "NORTE":"Regional 8 - Atlántico",
        "COSTA":"Regional 8 - Atlántico",
    }
    if "rg_descripcion" in df.columns:
        df["rg_descripcion"] = df["rg_descripcion"].str.upper().replace(rmap)

    # Blindaje numérico
    for col in df.columns:
        if any(x in col for x in ["ponal_","uri_","total","alerta",
                                   "capacidad","genero","extranjeros",
                                   "condenados","imputados"]):
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Columnas derivadas base
    df["_det_ponal"] = df["ponal_total_personas_salas"].astype(float)
    df["_det_uri"]   = df["uri_total_personas_salas"].astype(float)
    df["_td"]        = df["_det_ponal"] + df["_det_uri"]
    df["_tc"]        = df["ponal_capacidad_salas"].astype(float) + df["uri_capacidad_salas"].astype(float)
    df["_tcu"]       = df["ponal_personal_total"].astype(float)  + df["uri_personal_total"].astype(float)
    df["_hac"]       = (df["_td"] / df["_tc"].replace(0, np.nan) * 100).astype(float).round(1)

    if "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    if "fecha_reporte" in df.columns:
        df["fecha_reporte"] = pd.to_datetime(df["fecha_reporte"], errors="coerce")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def sg(df, col):
    return float(df[col].sum()) if col in df.columns else 0.0

def kcard(col, lbl, val, sub="", tipo=""):
    col.markdown(
        f"<div class='kpi-card {tipo}'>"
        f"<div class='kpi-lbl'>{lbl}</div>"
        f"<div class='kpi-val'>{val}</div>"
        f"<div class='kpi-sub'>{sub}</div></div>",
        unsafe_allow_html=True)

def bl(title="", h=430, margin=None):
    """Base layout Plotly. margin=dict(...) sobreescribe el default."""
    m = margin if margin is not None else dict(l=55, r=25, t=60, b=55)
    return dict(
        title_text=title, title_font_size=17, title_font_color="#0D2535",
        title_font_family="Barlow Condensed",
        font=dict(family="Barlow", size=14, color="#1E293B"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(245,248,252,1)",
        height=h, margin=m,
        legend=dict(font_size=14, title_font_size=14),
    )

def color_hac(val):
    if val >= 100: return "background-color:#DC2626;color:white;font-weight:bold"
    if val >= 90:  return "background-color:#EA580C;color:white;font-weight:bold"
    if val >= 80:  return "background-color:#EAB308;color:black;font-weight:bold"
    if val >= 70:  return "background-color:#22C55E;color:black"
    return "background-color:#166534;color:white"

def short_rg(name):
    """'Regional 8 - Llanos y Amazonia' -> 'Reg. 8'"""
    m = re.search(r'(\d+)', str(name))
    return f"Reg. {m.group(1)}" if m else str(name)[:8]


# ─────────────────────────────────────────────────────────────────────────────
# 4. FILTROS SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
def build_filters(df_raw):
    st.sidebar.markdown(
        "<h2 style='color:#38BDF8;font-family:Barlow Condensed;font-size:1.3rem;"
        "text-transform:uppercase;letter-spacing:.06em;border-bottom:1px solid #1E3A5F;"
        "padding-bottom:.4rem;margin-bottom:.6rem'>🔎 Filtros</h2>",
        unsafe_allow_html=True)

    tipo = st.sidebar.radio("Por instalación",
                            ["Todo (URI y PONAL)", "PONAL", "URI"])

    resps = ["Todos"] + sorted(df_raw["responsable"].dropna().unique().tolist())
    resp  = st.sidebar.selectbox("Responsable", resps)

    regs = sorted(df_raw["rg_descripcion"].dropna().unique()) if "rg_descripcion" in df_raw.columns else []
    f_reg = st.sidebar.multiselect("Regional (RG)", regs)

    unis = sorted(df_raw["unidad_codigo"].dropna().unique()) if "unidad_codigo" in df_raw.columns else []
    f_uni = st.sidebar.multiselect("Unidad", unis)

    salas = sorted(df_raw["sala_nombre"].dropna().unique()) if "sala_nombre" in df_raw.columns else []
    f_sala = st.sidebar.multiselect("Ubicación / Sala", salas)

    # ── Filtro Período Reporte (fecha_reporte de dim_reporte) ──
    if "fecha_reporte" in df_raw.columns and df_raw["fecha_reporte"].notna().any():
        fechas_disp = sorted(df_raw["fecha_reporte"].dropna().dt.date.unique())
        fechas_str  = [str(f) for f in fechas_disp]
        f_fechas = st.sidebar.multiselect("Período Reporte", fechas_str)
    else:
        f_fechas = []

    df = df_raw.copy()

    # Ajustar columnas según tipo de instalación
    if tipo == "PONAL":
        df["_td"]  = df["_det_ponal"]
        df["_tc"]  = df["ponal_capacidad_salas"].astype(float)
        df["_tcu"] = df["ponal_personal_total"].astype(float)
    elif tipo == "URI":
        df["_td"]  = df["_det_uri"]
        df["_tc"]  = df["uri_capacidad_salas"].astype(float)
        df["_tcu"] = df["uri_personal_total"].astype(float)

    if resp != "Todos":
        df = df[df["responsable"] == resp]
    if f_reg and "rg_descripcion" in df.columns:
        df = df[df["rg_descripcion"].isin(f_reg)]
    if f_uni and "unidad_codigo" in df.columns:
        df = df[df["unidad_codigo"].isin(f_uni)]
    if f_sala and "sala_nombre" in df.columns:
        df = df[df["sala_nombre"].isin(f_sala)]
    if f_fechas and "fecha_reporte" in df.columns:
        df = df[df["fecha_reporte"].dt.date.astype(str).isin(f_fechas)]

    # Recalcular hacinamiento tras filtros
    df["_hac"] = (df["_td"] / df["_tc"].replace(0, np.nan) * 100).astype(float).round(1)

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"<div style='font-size:.86rem;color:#94A3B8;'>📋 <b style='color:#38BDF8'>"
        f"{len(df):,}</b> registros activos</div>", unsafe_allow_html=True)

    # Créditos y autores
    st.sidebar.markdown(
        """
        <div class="footer-container" translate="no">
            <div class="footer-title">CONSTRUIDO POR:</div>
            <div class="footer-experts">Expertos en Data Science y Data Storytelling</div>
            <div class="footer-names">
                DIEGO HERNANDO MIRANDA<br>
                <span style="font-size: 18px;">&</span><br>
                JOSÉ DARÍO HERNÁNDEZ
            </div>
            <div class="footer-uni">
                Universidad Externado de Colombia<br>
                Maestría en Inteligencia de Negocios - Big Data<br>
                Bogotá, abril de 2026.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    return df, tipo


# ─────────────────────────────────────────────────────────────────────────────
# 5. TOTALIZADORES
# ─────────────────────────────────────────────────────────────────────────────
def render_totalizadores(df):
    n_inf  = df["id_reporte"].nunique()     if "id_reporte"    in df.columns else 0
    n_reg  = len(df)
    n_rg   = df["rg_descripcion"].nunique() if "rg_descripcion" in df.columns else 0
    n_uni  = df["unidad_codigo"].nunique()  if "unidad_codigo"  in df.columns else 0
    n_sala = df["sala_nombre"].nunique()    if "sala_nombre"    in df.columns else 0
    tot_det= int(df["_td"].sum())
    tot_cus= int(df["_tcu"].sum())

    items = [
        (f"{n_inf:,}",   "Informes"),
        (f"{n_reg:,}",   "Registros"),
        (f"{n_rg:,}",    "Regionales RG"),
        (f"{n_uni:,}",   "Unidades"),
        (f"{n_sala:,}",  "Estaciones/Salas"),
        (f"{tot_det:,}", "Total Detenidos"),
        (f"{tot_cus:,}", "Personal Custodia"),
    ]
    cols = st.columns(len(items))
    for col, (val, lbl) in zip(cols, items):
        col.markdown(
            f"<div class='tot-box'><div class='tot-num'>{val}</div>"
            f"<div class='tot-lbl'>{lbl}</div></div>",
            unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 6. MÉTRICAS ANALÍTICAS
# ─────────────────────────────────────────────────────────────────────────────
def render_metricas(df):
    td   = df["_td"].sum();   tcu  = df["_tcu"].sum(); tc = df["_tc"].sum()
    imp  = sg(df,"ponal_imputados") + sg(df,"uri_imputados")
    cond = sg(df,"ponal_condenados")+ sg(df,"uri_condenados")
    ext  = sg(df,"ponal_extranjeros")+sg(df,"uri_extranjeros")
    cpcar= sg(df,"ponal_casa_por_carcel")+sg(df,"uri_casa_por_carcel")
    det_p= sg(df,"ponal_total_personas_salas"); cap_p=sg(df,"ponal_capacidad_salas")
    det_u= sg(df,"uri_total_personas_salas");   cap_u=sg(df,"uri_capacidad_salas")

    rc    = round(td/tcu,2)           if tcu>0  else 0
    pcond = round(cond/td*100,1)      if td>0   else 0
    pimp  = round(imp/td*100,1)       if td>0   else 0
    cap_p_= round(det_p/cap_p*100,1)  if cap_p>0 else 0
    cap_u_= round(det_u/cap_u*100,1)  if cap_u>0 else 0
    pcp   = round(cpcar/td*100,1)     if td>0   else 0
    pext  = round(ext/td*100,1)       if td>0   else 0

    st.markdown("<div class='sec'>📐 Métricas Analíticas</div>", unsafe_allow_html=True)
    cols = st.columns(7)
    kcard(cols[0],"Ratio Custodio",    f"{rc}",      "det/custodio",
          "rojo" if rc>5 else ("naranja" if rc>3 else "verde"))
    kcard(cols[1],"% Condenados",      f"{pcond}%",  f"{int(cond):,} pers.")
    kcard(cols[2],"% Imputados",       f"{pimp}%",   f"{int(imp):,} pers.")
    kcard(cols[3],"Cap. Op. PONAL",    f"{cap_p_}%", "det/cap",
          "rojo" if cap_p_>=100 else ("naranja" if cap_p_>=80 else "verde"))
    kcard(cols[4],"Cap. Op. URI",      f"{cap_u_}%", "det/cap",
          "rojo" if cap_u_>=100 else ("naranja" if cap_u_>=80 else "verde"))
    kcard(cols[5],"Casa x Cárcel",     f"{pcp}%",    f"{int(cpcar):,} con medida",
          "rojo" if pcp>10 else "naranja")
    kcard(cols[6],"Ratio Extranjeros", f"{pext}%",   f"{int(ext):,} extranjeros",
          "naranja" if pext>5 else "")


# ─────────────────────────────────────────────────────────────────────────────
# 7. GRÁFICAS
# ─────────────────────────────────────────────────────────────────────────────

def chart_gauge(df):
    val   = float(df["_hac"].mean()) if df["_hac"].notna().any() else 0
    color = "#DC2626" if val>=100 else ("#EA580C" if val>=80 else "#16A34A")
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta", value=val,
        number={"suffix":"%","font":{"size":48,"family":"Barlow Condensed","color":color}},
        delta={"reference":100,"decreasing":{"color":"#16A34A"},"increasing":{"color":"#DC2626"}},
        gauge={"axis":{"range":[0,200],"tickfont":{"size":14}},
               "bar":{"color":color,"thickness":.28},"bgcolor":"white",
               "steps":[{"range":[0,80],"color":"#DCFCE7"},
                        {"range":[80,100],"color":"#FEF9C3"},
                        {"range":[100,200],"color":"#FEE2E2"}],
               "threshold":{"line":{"color":"#DC2626","width":4},"thickness":.85,"value":100}},
        title={"text":"Ocupación Global","font":{"size":17,"family":"Barlow Condensed","color":"#0D2535"}}))
    fig.update_layout(**bl(h=300, margin=dict(l=30,r=30,t=55,b=10)))
    return fig


def chart_treemap(df):
    td  = df["_td"].sum()
    grp = df.groupby(["rg_descripcion","sala_nombre"])["_td"].sum().reset_index()
    grp["% Part"] = (grp["_td"]/td*100).round(1)
    fig = px.treemap(grp, path=["rg_descripcion","sala_nombre"], values="_td",
                     custom_data=["% Part"], color="_td",
                     color_continuous_scale=px.colors.sequential.Blues)
    fig.update_traces(
        hovertemplate="<b>%{label}</b><br>Detenidos: %{value:,}<br>Participación: %{customdata[0]}%",
        textfont_size=15)
    fig.update_layout(**bl("🗺️ Jerarquía Regional — Distribución de Detenidos", h=460))
    return fig


def chart_hac_barras(df):
    """
    Eje X: etiqueta corta 'Reg. N' en la barra.
    Hover: nombre regional completo.
    Margen inferior ampliado para mostrar nombres completos bajo el eje.
    Rango Y extendido para que el valor encima de la barra no quede cortado.
    """
    grp = (df.groupby("rg_descripcion")["_hac"].mean()
             .sort_values(ascending=False).reset_index())
    grp.columns = ["Regional", "Hac"]
    grp["RegCorta"]    = grp["Regional"].apply(short_rg)
    grp["RegCompleta"] = grp["Regional"]

    clr = ["#DC2626" if v>=100 else ("#EA580C" if v>=80 else "#1565C0")
           for v in grp["Hac"]]

    fig = go.Figure(go.Bar(
        x=grp["RegCorta"],
        y=grp["Hac"],
        marker_color=clr,
        text=[f"<b>{v:.1f}%</b>" for v in grp["Hac"]],
        textposition="outside",
        textfont=dict(size=15, color="#0D2535"),
        customdata=grp["RegCompleta"],
        hovertemplate="<b>%{customdata}</b><br>Hacinamiento: <b>%{y:.1f}%</b><extra></extra>",
    ))

    fig.add_hline(y=100, line_dash="dash", line_color="#DC2626", line_width=2,
                  annotation_text="100% Capacidad",
                  annotation_font_size=12, annotation_font_color="#DC2626",
                  annotation_position="top right")

    max_val = grp["Hac"].max() if len(grp) > 0 else 200

    fig.update_layout(
        **bl("🔴 Hacinamiento Promedio por Regional", h=520,
             margin=dict(l=60, r=25, t=70, b=80)),
        yaxis_range=[0, max_val * 1.18],
    )
    fig.update_yaxes(ticksuffix="%", gridcolor="#E2E8F0",
                     title_text="% Ocupación", title_font_size=14)
    # Eje X: solo etiqueta corta "Reg. N" — hover muestra nombre completo
    fig.update_xaxes(
        tickmode="array",
        tickvals=grp["RegCorta"].tolist(),
        ticktext=[f"<b>{rc}</b>" for rc in grp["RegCorta"]],
        tickfont=dict(size=14, color="#0D2535", family="Barlow Condensed"),
        tickangle=0,
        automargin=True,
    )
    return fig


def chart_tiempo(df):
    if "mes_anio" not in df.columns or "fecha" not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="Sin datos temporales", showarrow=False)
        return fig
    ref = df.groupby("mes_anio")["fecha"].min()
    grp = (df.groupby("mes_anio")
             .agg(ponal=("_det_ponal","sum"), uri=("_det_uri","sum"))
             .reset_index().merge(ref.rename("f"), on="mes_anio").sort_values("f"))
    fig = go.Figure()
    fig.add_trace(go.Bar(x=grp["mes_anio"], y=grp["ponal"], name="PONAL",
                         marker_color="#1565C0", text=grp["ponal"].astype(int),
                         textposition="outside", textfont_size=12))
    fig.add_trace(go.Bar(x=grp["mes_anio"], y=grp["uri"], name="URI",
                         marker_color="#DC2626", text=grp["uri"].astype(int),
                         textposition="outside", textfont_size=12))
    fig.update_layout(barmode="group",
                      **bl("📈 Comportamiento en el Tiempo — PONAL vs URI (15 meses)", h=440))
    fig.update_yaxes(title_text="Personas Detenidas", title_font_size=14, gridcolor="#E2E8F0")
    fig.update_xaxes(tickfont_size=13)
    return fig


def chart_extranjeros(df):
    """
    Doble pie PONAL / URI.
    Margen superior ampliado para separar título principal de subtítulos de subplots.
    """
    ep = sg(df,"ponal_extranjeros"); eu = sg(df,"uri_extranjeros")
    dp = sg(df,"ponal_total_personas_salas"); du = sg(df,"uri_total_personas_salas")

    # Sin subplot_titles para evitar solapamiento — subtítulos como anotaciones manuales
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type":"domain"},{"type":"domain"}]],
        horizontal_spacing=0.06,
    )
    for c, ext, det, nm in [(1,ep,dp,"PONAL"),(2,eu,du,"URI")]:
        nac = max(det-ext, 0)
        fig.add_trace(go.Pie(
            labels=["Extranjeros","Nacionales"],
            values=[max(ext,0), nac],
            hole=.42, name=nm,
            marker=dict(colors=["#DC2626","#1565C0"]),
            textinfo="label+percent",
            textfont=dict(size=14),
            insidetextorientation="radial",
        ), row=1, col=c)

    fig.update_layout(
        **bl("🌎 Presión Migratoria: Extranjeros por Instalación",
             h=460, margin=dict(l=30, r=150, t=80, b=30)),
    )
    # Subtítulos manuales posicionados como texto limpio sobre cada anillo
    for x_pos, label in [(0.22, "PONAL"), (0.78, "URI")]:
        fig.add_annotation(
            x=x_pos, y=1.06, xref="paper", yref="paper",
            text=f"<b>{label}</b>",
            showarrow=False,
            font=dict(size=18, color="#0D2535", family="Barlow Condensed"),
            align="center",
        )
    return fig


def chart_alimentacion(df):
    vals   = [sg(df,"ponal_alim_alcaldia"), sg(df,"ponal_alim_policia"),
              sg(df,"ponal_alim_inpec"),    sg(df,"ponal_alim_familiares")]
    labels = ["Alcaldía","Policía","INPEC","Familiares"]
    fig = go.Figure(go.Pie(
        labels=labels, values=vals, hole=.5,
        marker=dict(colors=["#1565C0","#16A34A","#EA580C","#7C3AED"]),
        textinfo="label+percent+value", textfont_size=15,
        hovertemplate="<b>%{label}</b><br>%{value:,.0f} personas<br>%{percent}<extra></extra>"))
    fig.update_layout(**bl("🍽️ Responsable de Alimentación de los Detenidos", h=400))
    return fig


def chart_personal(df):
    vals   = [sg(df,"ponal_personal_of") +sg(df,"uri_personal_of"),
              sg(df,"ponal_personal_me") +sg(df,"uri_personal_me"),
              sg(df,"ponal_personal_ptag")+sg(df,"uri_personal_ptag"),
              sg(df,"ponal_personal_aux")+sg(df,"uri_personal_aux")]
    labels = ["Oficial (OF)","Mando Ejec. (M/E)","Patrulleros (PT/AG)","Auxiliares (AUX)"]
    fig = go.Figure(go.Pie(
        labels=labels, values=vals, hole=.4,
        marker=dict(colors=["#0D2535","#1565C0","#38BDF8","#94A3B8"]),
        textinfo="label+percent", textfont_size=15,
        hovertemplate="<b>%{label}</b><br>%{value:,.0f} efectivos<br>%{percent}<extra></extra>"))
    fig.update_layout(**bl("👮 Distribución del Personal de Custodia por Rango", h=400))
    return fig


def chart_juridico(df):
    """
    Condenados vs Imputados por regional.
    Eje X: etiqueta corta 'Reg. N' + hover con nombre completo.
    Ancho completo, barras grandes.
    """
    gc  = "rg_descripcion" if "rg_descripcion" in df.columns else "sala_nombre"
    grp = df.groupby(gc)[["ponal_condenados","ponal_imputados"]].sum().reset_index()
    grp.columns = [gc, "Condenados", "Imputados"]
    grp = grp.sort_values("Imputados", ascending=False)

    if gc == "rg_descripcion":
        eje_x    = grp[gc].apply(short_rg).tolist()
        hover_nm = grp[gc].tolist()
    else:
        eje_x    = grp[gc].tolist()
        hover_nm = grp[gc].tolist()

    fig = go.Figure()
    for col, nm, clr in [("Condenados","Condenados","#DC2626"),
                          ("Imputados", "Imputados", "#1565C0")]:
        fig.add_trace(go.Bar(
            name=nm, x=eje_x, y=grp[col],
            marker_color=clr,
            text=grp[col].astype(int),
            textposition="outside",
            textfont=dict(size=14),
            customdata=hover_nm,
            hovertemplate="<b>%{customdata}</b><br>" + nm + ": %{y:,}<extra></extra>",
        ))

    max_val = max(grp["Condenados"].max(), grp["Imputados"].max()) if len(grp) else 1
    fig.update_layout(
        barmode="group",
        **bl("⚖️ Condenados vs Imputados por Regional", h=500,
             margin=dict(l=65, r=25, t=70, b=70)),
        yaxis_range=[0, max_val * 1.18],
    )
    fig.update_yaxes(title_text="Personas", title_font_size=15,
                     gridcolor="#E2E8F0", tickfont_size=13)
    fig.update_xaxes(tickfont=dict(size=16, color="#0D2535"), tickangle=0)
    return fig


def chart_imputados12(df):
    """
    Imputados >12 meses por unidad / regional.
    Eje X: etiqueta corta si es regional + hover completo.
    Ancho completo, barras grandes.
    """
    gc  = "sala_nombre" if "sala_nombre" in df.columns else "rg_descripcion"
    grp = (df.groupby(gc)
             .agg(tot=("_td","sum"), imp=("ponal_imputados","sum"),
                  m12=("ponal_imputados_mas_12_meses","sum"))
             .reset_index().nlargest(12,"m12").sort_values("m12", ascending=False))

    if gc == "rg_descripcion":
        eje_x    = grp[gc].apply(short_rg).tolist()
        hover_nm = grp[gc].tolist()
    else:
        eje_x    = grp[gc].tolist()
        hover_nm = grp[gc].tolist()

    fig = go.Figure()
    for col, nm, clr in [("tot","Total Detenidos","#94A3B8"),
                          ("imp","Imputados",       "#1565C0"),
                          ("m12",">12 Meses",       "#DC2626")]:
        fig.add_trace(go.Bar(
            name=nm, x=eje_x, y=grp[col].astype(float),
            marker_color=clr,
            text=grp[col].astype(int),
            textposition="outside",
            textfont=dict(size=13),
            customdata=hover_nm,
            hovertemplate="<b>%{customdata}</b><br>" + nm + ": %{y:,}<extra></extra>",
        ))

    max_val = grp["tot"].max() if len(grp) else 1
    fig.update_layout(
        barmode="group",
        **bl("⏰ Imputados >12 Meses — Comparativo por Unidad", h=500,
             margin=dict(l=65, r=25, t=70, b=70)),
        yaxis_range=[0, max_val * 1.18],
    )
    fig.update_yaxes(title_text="Personas", title_font_size=15,
                     gridcolor="#E2E8F0", tickfont_size=13)
    fig.update_xaxes(tickfont=dict(size=14, color="#0D2535"), tickangle=0)
    return fig


def chart_efecto_admin(df):
    gc  = "sala_nombre" if "sala_nombre" in df.columns else "rg_descripcion"
    grp = (df.groupby(gc)
             .agg(sd=("ponal_sin_documentacion","sum"), td=("_td","sum"))
             .reset_index())
    grp = grp[grp["sd"] > 0].nlargest(12,"sd").sort_values("sd")
    grp["pct"] = (grp["sd"] / grp["td"].replace(0, np.nan) * 100).round(1)

    if gc == "rg_descripcion":
        eje_y    = grp[gc].apply(short_rg).tolist()
        hover_nm = grp[gc].tolist()
    else:
        eje_y    = grp[gc].tolist()
        hover_nm = grp[gc].tolist()

    fig = go.Figure(go.Bar(
        x=grp["sd"].astype(float), y=eje_y, orientation="h",
        marker_color="#7C3AED",
        text=[f"{int(v):,}  ({p}%)" for v,p in zip(grp["sd"],grp["pct"])],
        textposition="outside", textfont_size=13,
        customdata=hover_nm,
        hovertemplate="<b>%{customdata}</b><br>Sin doc: %{x:,}<extra></extra>",
    ))
    fig.update_layout(**bl("📋 Efecto Administrativo: Sin Documentación para Traslado",
                           h=460, margin=dict(l=90, r=60, t=70, b=55)))
    fig.update_xaxes(title_text="Personas sin documento", title_font_size=14)
    fig.update_yaxes(tickfont_size=13)
    return fig


def chart_genero(df):
    """
    Barras simples — sin marcadores/diamantes adicionales.
    Etiqueta de texto incluye valor absoluto + porcentaje.
    Altura generosa para que no solape con la gráfica de extranjeros debajo.
    """
    gm  = sg(df,"ponal_genero_m")    + sg(df,"uri_genero_m")
    gf  = sg(df,"ponal_genero_f")    + sg(df,"uri_genero_f")
    glg = sg(df,"ponal_genero_lgbti")+ sg(df,"uri_genero_lgbti")

    total_g = gm + gf + glg if (gm + gf + glg) > 0 else 1
    vals    = [gm, gf, glg]
    labels  = ["Masculino", "Femenino", "LGBTI+"]
    colores = ["#1565C0", "#DB2777", "#7C3AED"]
    pcts    = [round(v/total_g*100, 1) for v in vals]

    fig = go.Figure(go.Bar(
        x=labels,
        y=vals,
        marker_color=colores,
        text=[f"<b>{int(v):,}</b>  ({p}%)" for v, p in zip(vals, pcts)],
        textposition="outside",
        textfont=dict(size=16, family="Barlow Condensed"),
        hovertemplate="<b>%{x}</b><br>Personas: %{y:,}<extra></extra>",
        width=0.45,
    ))

    max_val = max(vals) if any(v > 0 for v in vals) else 1
    fig.update_layout(
        **bl("👥 Distribución por Género", h=460,
             margin=dict(l=65, r=25, t=70, b=60)),
        yaxis_range=[0, max_val * 1.22],
        showlegend=False,
    )
    fig.update_yaxes(title_text="Personas", title_font_size=14,
                     gridcolor="#E2E8F0", tickfont_size=13)
    fig.update_xaxes(tickfont=dict(size=17, color="#0D2535"))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 8. TABLA RANKING ALERTAS — sum() original, sin mediana
# ─────────────────────────────────────────────────────────────────────────────
def render_tabla_alertas(df, td):
    gc   = "sala_nombre" if "sala_nombre" in df.columns else "rg_descripcion"
    aggs = {"td":("_td","sum"),"tc":("_tc","sum"),"hac":("_hac","mean")}
    if "rg_descripcion" in df.columns:
        aggs["rg"] = ("rg_descripcion","first")
    grp  = df.groupby(gc).agg(**aggs).reset_index()
    grp["% Part"] = (grp["td"] / td * 100).round(2)
    grp  = grp.rename(columns={
        "td":"Detenidos", "tc":"Capacidad", "hac":"% Hacinamiento",
        "rg":"Regional",  "% Part":"% Particip.", gc:"Unidad/Sala"
    })
    show = (["Regional","Unidad/Sala","Detenidos","Capacidad","% Hacinamiento","% Particip."]
            if "Regional" in grp.columns
            else ["Unidad/Sala","Detenidos","Capacidad","% Hacinamiento","% Particip."])
    st.dataframe(
        grp[show].sort_values("% Hacinamiento", ascending=False)
                 .style.applymap(color_hac, subset=["% Hacinamiento"])
                 .format({"% Hacinamiento":"{:.1f}%","% Particip.":"{:.2f}%",
                          "Detenidos":"{:,.0f}","Capacidad":"{:,.0f}"}),
        use_container_width=True, height=560)


# ─────────────────────────────────────────────────────────────────────────────
# 9. MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    inject_styles()
    df_raw   = load_data()
    df, tipo = build_filters(df_raw)

    if df.empty:
        st.warning("⚠️ Sin datos para los filtros seleccionados.")
        st.stop()

    td      = df["_td"].sum()
    hac_avg = float(df["_hac"].mean()) if df["_hac"].notna().any() else 0
    badge   = ("🔴 HACINAMIENTO CRÍTICO" if hac_avg>=100
               else ("🟠 ATENCIÓN" if hac_avg>=80 else "🟢 CONTROLADO"))

    # ── Header ──
    st.markdown(
        f"<div class='main-header'>"
        f"<h1>🏛️ Sistema de Inteligencia Penitenciaria v4.1</h1>"
        f"<p>Gestión de Detenidos · Vista: <b>{tipo}</b> · {badge}</p>"
        f"</div>", unsafe_allow_html=True)

    # ── Totalizadores ──
    st.markdown("<div class='sec'>🔢 Totalizadores</div>", unsafe_allow_html=True)
    render_totalizadores(df)

    # ── Métricas ──
    render_metricas(df)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ──
    t1, t2, t3, t4, t5 = st.tabs([
        "🔍 CONTROL OPERATIVO",
        "⚖️ ESTATUS JURÍDICO",
        "👥 PERFIL POBLACIONAL",
        "🍽️ SERVICIOS Y CUSTODIA",
        "🚨 CENTRO DE ALERTAS",
    ])

    # ─── TAB 1 ───────────────────────────────────────────────────────────────
    with t1:
        c1, c2 = st.columns([.55, .45])
        with c1:
            st.plotly_chart(chart_treemap(df),    use_container_width=True)
        with c2:
            st.plotly_chart(chart_gauge(df),      use_container_width=True)
            st.plotly_chart(chart_hac_barras(df), use_container_width=True)
        st.plotly_chart(chart_tiempo(df), use_container_width=True)

    # ─── TAB 2 — 3 gráficas apiladas a ancho completo ────────────────────────
    with t2:
        imp  = sg(df,"ponal_imputados")+sg(df,"uri_imputados")
        m12  = sg(df,"ponal_imputados_mas_12_meses")+sg(df,"uri_imputados_mas_12_meses")
        cond = sg(df,"ponal_condenados")+sg(df,"uri_condenados")
        sdoc = sg(df,"ponal_sin_documentacion")
        pct12= round(m12/imp*100,1) if imp>0 else 0

        mc = st.columns(4)
        kcard(mc[0],"Imputados >12M",   f"{int(m12):,}",  f"{pct12}% de imputados","rojo")
        kcard(mc[1],"Total Imputados",  f"{int(imp):,}",   f"{round(imp/td*100,1)}% del total")
        kcard(mc[2],"Total Condenados", f"{int(cond):,}",  f"{round(cond/td*100,1)}% del total")
        kcard(mc[3],"Sin Doc Traslado", f"{int(sdoc):,}",  "efecto administrativo","naranja")

        st.markdown("<br>", unsafe_allow_html=True)
        # Las 3 gráficas a ancho completo, apiladas verticalmente
        st.plotly_chart(chart_juridico(df),     use_container_width=True)
        st.plotly_chart(chart_imputados12(df),  use_container_width=True)
        st.plotly_chart(chart_efecto_admin(df), use_container_width=True)

    # ─── TAB 3 ───────────────────────────────────────────────────────────────
    with t3:
        gm  = sg(df,"ponal_genero_m")    +sg(df,"uri_genero_m")
        gf  = sg(df,"ponal_genero_f")    +sg(df,"uri_genero_f")
        glg = sg(df,"ponal_genero_lgbti")+sg(df,"uri_genero_lgbti")
        ext = sg(df,"ponal_extranjeros") +sg(df,"uri_extranjeros")
        ven = sg(df,"ponal_venezolanos") +sg(df,"uri_venezolanos")
        cmed= sg(df,"ponal_condiciones_medicas")+sg(df,"uri_condiciones_medicas")

        mc = st.columns(4)
        kcard(mc[0],"Masculino",     f"{int(gm):,}",   f"{round(gm/td*100,1)}% del total")
        kcard(mc[1],"Femenino",      f"{int(gf):,}",   f"{round(gf/td*100,1)}% del total")
        kcard(mc[2],"LGBTI+",        f"{int(glg):,}",  f"{round(glg/td*100,1)}% del total")
        kcard(mc[3],"Cond. Médicas", f"{int(cmed):,}", f"{round(cmed/td*100,1)}% del total","naranja")

        st.markdown("<br>", unsafe_allow_html=True)
        # Apiladas verticalmente — sin solapamiento entre género y extranjeros
        st.plotly_chart(chart_genero(df),      use_container_width=True)
        st.plotly_chart(chart_extranjeros(df), use_container_width=True)

        pct_ven = round(ven/ext*100,1) if ext>0 else 0
        st.info(f"🇻🇪 **Venezolanos:** {int(ven):,} ({pct_ven}% de los extranjeros) "
                f"| **Total Extranjeros:** {int(ext):,} ({round(ext/td*100,1)}% del total)")

    # ─── TAB 4 ───────────────────────────────────────────────────────────────
    with t4:
        tcu  = df["_tcu"].sum()
        of   = sg(df,"ponal_personal_of") +sg(df,"uri_personal_of")
        me   = sg(df,"ponal_personal_me") +sg(df,"uri_personal_me")
        ptag = sg(df,"ponal_personal_ptag")+sg(df,"uri_personal_ptag")
        aux  = sg(df,"ponal_personal_aux")+sg(df,"uri_personal_aux")
        rc   = round(td/tcu,2) if tcu>0 else 0

        mc = st.columns(5)
        kcard(mc[0],"Total Custodia",      f"{int(tcu):,}", "efectivos totales")
        kcard(mc[1],"Oficial (OF)",        f"{int(of):,}",  f"{round(of/tcu*100,1) if tcu>0 else 0}%")
        kcard(mc[2],"Mando Ejec. (M/E)",  f"{int(me):,}",  f"{round(me/tcu*100,1) if tcu>0 else 0}%")
        kcard(mc[3],"Patrulleros (PT/AG)",f"{int(ptag):,}", f"{round(ptag/tcu*100,1) if tcu>0 else 0}%")
        kcard(mc[4],"Ratio Custodia",     f"{rc}",          "det/custodio",
              "rojo" if rc>5 else ("naranja" if rc>3 else "verde"))

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(chart_personal(df),     use_container_width=True)
        with c2:
            st.plotly_chart(chart_alimentacion(df), use_container_width=True)

    # ─── TAB 5 ───────────────────────────────────────────────────────────────
    with t5:
        mas36 = sg(df,"ponal_mas_36_horas")+sg(df,"uri_mas_36_horas")
        cpcar = sg(df,"ponal_casa_por_carcel")+sg(df,"uri_casa_por_carcel")
        hac_cr= int((df["_hac"]>=100).sum())

        mc = st.columns(4)
        kcard(mc[0],"🔴 Alertas >36h",          f"{int(mas36):,}",f"{round(mas36/td*100,1)}% del total","rojo")
        kcard(mc[1],"🏠 Casa x Cárcel",          f"{int(cpcar):,}",f"{round(cpcar/td*100,1)}% del total","naranja")
        kcard(mc[2],"🚨 Registros Hacin.≥100%",  f"{hac_cr:,}",   "en situación crítica","rojo")
        kcard(mc[3],"📊 Hac. Promedio",           f"{hac_avg:.1f}%","ocupación media")

        st.markdown("<div class='sec'>🚩 Ranking de Unidades / Estaciones por Hacinamiento</div>",
                    unsafe_allow_html=True)
        render_tabla_alertas(df, td)
        st.download_button("⬇️ Exportar datos filtrados (.csv)",
                           df.to_csv(index=False).encode("utf-8"),
                           "detenidos_filtrado.csv","text/csv")

    st.markdown(
        "<div class='footer'>Sistema de Inteligencia Penitenciaria v4.1 · "
        "DWH Institucional · Dashboard Ejecutivo Stakeholders</div>",
        unsafe_allow_html=True)


if __name__ == "__main__":
    main()

# ─────────────────────────────────────────────────────────────────────────────
# EJECUTAR CON: poetry run streamlit run app.py
# ─────────────────────────────────────────────────────────────────────────────