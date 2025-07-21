import pandas as pd
import numpy as np
import dash
from dash import html, dcc, Input, Output, callback
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from dash import dash_table

suppress_callback_exceptions=True

# 📥 Carga de datos
jugadores_df = pd.read_excel("data/Jugadores1A_2024.xlsx")
jugadores_df.replace("-", np.nan, inplace=True)

# Simular columna "Potencial" si no existe
if "Potencial" not in jugadores_df.columns:
    jugadores_df["Potencial"] = np.random.choice(["Alto", "Medio", "Bajo"], size=len(jugadores_df))

# Variables para entrenamiento
columnas_numericas = [
    "Duelos ganados %",
    "Duelos defensivos ganados %",
    "Duelos aéreos ganados %",
    "Precisión centros %",
    "Regates realizados %",
    "Duelos atacantes ganados %",
    "Precisión pases %",
    "Precisión pases hacia adelante %",
    "Precisión pases hacia atrás %",
    "Precisión pases largos %",
    "Precisión pases en el último tercio %"
]

# Posición y modelo por defecto
posicion_default = sorted(jugadores_df["Posición principal"].dropna().unique())[0]
modelo_default = "rf"

# Heatmap
heatmap_data = (
    jugadores_df.groupby(["Posición principal", "Potencial"])
    .size()
    .unstack(fill_value=0)
    .reindex(columns=["Alto", "Medio", "Bajo"])
)

heatmap_fig = px.imshow(
    heatmap_data,
    labels=dict(x="Potencial", y="Posición", color="Cantidad"),
    color_continuous_scale="Blues",
    text_auto=True,
    aspect="auto"
)
heatmap_fig.update_layout(
    title="🎯 Distribución de Potencial por Posición",
    width=1200,
    height=800,
    paper_bgcolor="#121212",
    plot_bgcolor="#121212",
    font_color="white",
    margin=dict(t=80, l=150, r=80, b=100),
    xaxis=dict(tickangle=0, tickfont=dict(size=16), automargin=True),
    yaxis=dict(tickfont=dict(size=16), automargin=True)
)

layout = dbc.Container([
    html.H2(
        "📈 Equipos Campeonato Itau 2024 - Módulo Machine Learning",
        style={"textAlign": "center", "color": "#ffffff", "fontSize": "32px", "marginBottom": "30px"}
    ),

    html.H2("📊 Análisis de Potencial con Machine Learning", className="my-3"),

    dcc.Graph(figure=heatmap_fig, style={"marginBottom": "40px"}),

    html.H4("🤖 Selecciona el Modelo de Aprendizaje"),
    dcc.Dropdown(
        id="dropdown-modelo-ml",
        options=[
            {"label": "Random Forest", "value": "rf"},
            {"label": "Regresión Logística", "value": "lr"}
        ],
        value=modelo_default,
        className="mb-4"
    ),

    html.H4("🎯 Entrenamiento por posición"),
    dcc.Dropdown(
        id="dropdown-posicion-ml",
        options=[{"label": pos, "value": pos} for pos in sorted(jugadores_df["Posición principal"].dropna().unique())],
        value=posicion_default,
        placeholder="Selecciona una posición",
        className="mb-3"
    ),

    dbc.Row([
        dbc.Col(dcc.Graph(id="grafico-importancia-ml"), md=6),
        dbc.Col(dash_table.DataTable(
            id="tabla-metricas-ml",
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "center"},
            style_header={"backgroundColor": "#222", "color": "white"},
            style_data={
                "backgroundColor": "#1a1a1a",
                "color": "white",
                "border": "1px solid white"
            }
        ), md=6),
    ])
], fluid=True)

@callback(
    Output("grafico-importancia-ml", "figure"),
    Output("tabla-metricas-ml", "data"),
    Output("tabla-metricas-ml", "columns"),
    Input("dropdown-posicion-ml", "value"),
    Input("dropdown-modelo-ml", "value")
)
def entrenar_modelo(posicion, modelo_tipo):
    if not posicion:
        posicion = posicion_default
    if not modelo_tipo:
        modelo_tipo = modelo_default

    df_pos = jugadores_df[jugadores_df["Posición principal"] == posicion].copy()

    if df_pos.empty:
        return go.Figure(), [], []

    try:
        X = df_pos[columnas_numericas].fillna(0)
        y = df_pos["Potencial"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        if modelo_tipo == "rf":
            model = RandomForestClassifier(random_state=42)
            importancias = model.fit(X_train, y_train).feature_importances_
        elif modelo_tipo == "lr":
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            importancias = abs(model.coef_[0])
        else:
            return go.Figure(), [], []

        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)

        fig = px.bar(
            x=importancias,
            y=columnas_numericas,
            orientation="h",
            labels={"x": "Importancia", "y": "Variable"},
            title=f"🔍 Variables más influyentes con {modelo_tipo.upper()} en posición: {posicion}",
            color=importancias,
            color_continuous_scale="Blues"
        )
        fig.update_layout(
            paper_bgcolor="#121212",
            plot_bgcolor="#121212",
            font_color="white",
            yaxis={"categoryorder": "total ascending"},
            height=500
        )

        metricas = ["precision", "recall", "f1-score"]
        data = [
            {
                "Clase": clase,
                **{m.capitalize(): round(report[clase][m], 2) for m in metricas}
            }
            for clase in report if clase in ["Alto", "Medio", "Bajo"]
        ]
        columns = [{"name": col, "id": col} for col in data[0].keys()] if data else []

        return fig, data, columns

    except Exception as e:
        print(f"❌ Error: {e}")
        return go.Figure(), [], []