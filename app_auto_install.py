import subprocess
import sys

# Lista de paquetes necesarios
paquetes = [
    "dash",
    "dash-bootstrap-components",
    "pandas",
    "numpy",
    "scikit-learn",
    "openpyxl",
    "plotly"
]

# Instalar cada paquete si no está presente
for paquete in paquetes:
    try:
        __import__(paquete.replace("-", "_"))
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", paquete])

# Código principal de la app
from dash import Dash
import ml_modificado_corregido_completo as ml

external_stylesheets = [
    "https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css"
]

app = Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = ml.layout

server = app.server  # Para despliegue

if __name__ == "__main__":
    app.run(debug=True)