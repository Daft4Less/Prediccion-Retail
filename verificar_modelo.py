import pandas as pd
import json

try:
    with open('modelo/metadata.json', 'r') as f:
        metadata = json.load(f)
    df = pd.read_parquet('modelo/historico_features.parquet')
    
    print("Modelo cargado correctamente")
    print("Tipo de modelo: ModeloONNX")
    print(f"Error de validacion (MAE): {metadata['mae']:.2f} unidades")
    print(f"Registros del historico usado: {len(df)}")
    print(f"Variables que usa el modelo: {len(metadata['columnas_features'])}")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
