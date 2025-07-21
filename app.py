import subprocess
import sys

##########################
def instalar_paquete(nombre):
    subprocess.check_call([sys.executable, "-m", "pip", "install", nombre])

##############################
instalar_paquete("dash")
instalar_paquete("pandas")
instalar_paquete("scikit-learn")
instalar_paquete("dash-bootstrap-components")
instalar_paquete("openpyxl")
instalar_paquete("plotly")

###########################
from dash import Dash
import NW_ml as ml  # Usa el archivo corregido

external_stylesheets = [
    "https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css"
]

app = Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = ml.layout

server = app.server  # Necesario para Render u otros servicios WSGI

if __name__ == "__main__":
  ##  app.run_server(debug=True)
    app.run(debug=True)