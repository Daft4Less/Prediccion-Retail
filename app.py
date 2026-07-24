"""
Sistema de Predicción de Demanda y Gestión de Inventario — IA para Retail
Aplicación construida con Streamlit sobre el modelo XGBoost ya entrenado y validado
en el notebook Prototipo_V2_Retail.ipynb.
"""

import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import plotly.graph_objects as go
import warnings
import textwrap
import logging

warnings.filterwarnings("ignore")

import onnxruntime as rt

# Configurar el sistema de logging para imprimir en la consola CMD con marcas de tiempo
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logging.info("Iniciando aplicación de predicción de retail...")


class ModeloONNX:
    """Envoltorio simple para que el modelo cargado desde .onnx se use exactamente
    igual que un modelo de scikit-learn/XGBoost: con un método .predict(X)."""
    def __init__(self, ruta_onnx):
        self.sesion = rt.InferenceSession(ruta_onnx)

    def predict(self, X):
        X_np = np.asarray(X, dtype=np.float32)
        return self.sesion.run(None, {"float_input": X_np})[0].flatten()

st.set_page_config(
    page_title="IA para Retail — Gestión de Inventario Predictivo",
    page_icon=":material/inventory_2:",
    layout="wide",
)

# =========================================================
# ESTILO VISUAL (tipografía + refinamiento de componentes)
# =========================================================
st.markdown(
    textwrap.dedent(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        /* Aplicar la fuente Inter a toda la app */
        html, body, [class*="st-"] {
            font-family: 'Inter', sans-serif;
        }

        /* Ocultar por completo la barra lateral (sidebar) y todos sus controles de colapso/expandir */
        [data-testid="stSidebar"],
        [data-testid="stSidebarCollapsedControl"],
        section[data-testid="stSidebar"],
        button[data-testid*="collapse" i],
        button[data-testid*="control" i],
        button[class*="collapse" i],
        button[class*="control" i],
        [data-testid="collapsedControl"] {
            display: none !important;
        }

        /* Configurar el contenedor principal para que sea centrado y use el espacio de forma óptima a pantalla completa */
        .main .block-container {
            max-width: 1200px !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
            padding-top: 2rem !important;
            margin-left: auto !important;
            margin-right: auto !important;
        }

        /* Fondo principal de la app */
        .stApp {
            background-color: #F8FAFC !important;
        }

        /* Rediseñar Contenedores st.container(border=True) como Tarjetas SaaS Blancas */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border: 1px solid #E2E8F0 !important;
            border-radius: 16px !important;
            background-color: #FFFFFF !important;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03) !important;
            padding: 24px !important;
        }

        /* Estilo para las Pestañas (Tabs) */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            border-bottom: 1px solid #E2E8F0;
        }
        .stTabs [data-baseweb="tab"] {
            font-family: 'Inter', sans-serif;
            font-weight: 600;
            color: #64748B;
            padding: 0.6rem 1.2rem;
            transition: color 0.15s ease;
        }
        .stTabs [data-baseweb="tab"]:hover {
            color: #2563EB;
        }
        .stTabs [aria-selected="true"] {
            color: #1E293B !important;
            border-bottom: 2px solid #2563EB !important;
        }

        /* Estilo General para Botones */
        .stButton>button {
            border-radius: 8px !important;
            font-weight: 600 !important;
            border: none !important;
            background: #2563EB !important; /* Azul por defecto */
            color: #FFFFFF !important;
            padding: 0.6rem 1.5rem !important;
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2) !important;
            transition: all 0.2s ease !important;
        }
        .stButton>button:hover {
            box-shadow: 0 6px 18px rgba(37, 99, 235, 0.4) !important;
            transform: translateY(-1px);
            color: #FFFFFF !important;
        }
        .stButton>button:active {
            transform: translateY(0);
        }

        /* Clase especial para envolver el botón de consulta general y hacerlo púrpura */
        .tab-general-btn div.stButton > button {
            background: #8B5CF6 !important;
            box-shadow: 0 4px 12px rgba(139, 92, 246, 0.2) !important;
        }
        .tab-general-btn div.stButton > button:hover {
            box-shadow: 0 6px 18px rgba(139, 92, 246, 0.4) !important;
            color: #FFFFFF !important;
        }

        /* Separadores de línea */
        hr {
            border-color: #E2E8F0;
        }

        /* Barra lateral */
        [data-testid="stSidebar"] {
            background-color: #F8FAFC !important;
            border-right: 1px solid #E2E8F0 !important;
        }
        /* Forzar visibilidad de textos e inputs en la barra lateral */
        [data-testid="stSidebar"] h1, 
        [data-testid="stSidebar"] h2, 
        [data-testid="stSidebar"] h3, 
        [data-testid="stSidebar"] h4, 
        [data-testid="stSidebar"] h5, 
        [data-testid="stSidebar"] h6,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] div {
            color: #1E293B !important;
        }
        [data-testid="stSidebar"] .stCaption {
            color: #64748B !important;
        }

        /* --- Visibilidad y contraste de textos en la página principal --- */
        /* Etiquetas de controles (Selectbox, Inputs, etc.) */
        [data-testid="stWidgetLabel"] p, 
        [data-testid="stWidgetLabel"] span,
        label.st-ae,
        .stSelectbox label,
        .stDateInput label,
        .stNumberInput label {
            color: #1E293B !important;
            font-weight: 500 !important;
        }
        /* Contenido seleccionado en selectboxes */
        div[data-baseweb="select"] div {
            color: #1E293B !important;
        }
        /* Contenido de inputs de texto/fecha/número */
        div[data-testid="stDateInput"] input,
        div[data-testid="stNumberInput"] input,
        input {
            color: #1E293B !important;
        }

        /* --- Segmented Control (Píldoras Fecha Única / Rango) --- */
        div[data-testid="stSegmentedControl"] {
            background-color: #F1F5F9 !important;
            padding: 4px !important;
            border-radius: 8px !important;
            border: 1px solid #E2E8F0 !important;
            display: inline-flex !important;
        }
        div[data-testid="stSegmentedControl"] button {
            background-color: transparent !important;
            border-radius: 6px !important;
            border: none !important;
            font-weight: 600 !important;
            padding: 6px 16px !important;
            transition: all 0.2s ease !important;
        }
        div[data-testid="stSegmentedControl"] button,
        div[data-testid="stSegmentedControl"] button p,
        div[data-testid="stSegmentedControl"] button span,
        div[data-testid="stSegmentedControl"] button div {
            color: #64748B !important;
        }
        div[data-testid="stSegmentedControl"] button:hover {
            background-color: #E2E8F0 !important;
        }
        div[data-testid="stSegmentedControl"] button:hover,
        div[data-testid="stSegmentedControl"] button:hover p,
        div[data-testid="stSegmentedControl"] button:hover span,
        div[data-testid="stSegmentedControl"] button:hover div {
            color: #1E293B !important;
        }
        div[data-testid="stSegmentedControl"] button[aria-checked="true"] {
            background-color: #FFFFFF !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
        }
        div[data-testid="stSegmentedControl"] button[aria-checked="true"],
        div[data-testid="stSegmentedControl"] button[aria-checked="true"] p,
        div[data-testid="stSegmentedControl"] button[aria-checked="true"] span,
        div[data-testid="stSegmentedControl"] button[aria-checked="true"] div {
            color: #2563EB !important;
            font-weight: 600 !important;
        }
        div[data-testid="stSegmentedControl"] button[aria-checked="true"]:hover {
            background-color: #FFFFFF !important;
            box-shadow: 0 1px 4px rgba(0,0,0,0.15) !important;
        }
        div[data-testid="stSegmentedControl"] button[aria-checked="true"]:hover,
        div[data-testid="stSegmentedControl"] button[aria-checked="true"]:hover p,
        div[data-testid="stSegmentedControl"] button[aria-checked="true"]:hover span,
        div[data-testid="stSegmentedControl"] button[aria-checked="true"]:hover div {
            color: #1D4ED8 !important;
        }

        /* Personalización del botón del cargador de archivos (st.file_uploader) */
        [data-testid="stFileUploader"] button {
            background-color: #FFFFFF !important;
            color: #475569 !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 6px !important;
            padding: 6px 12px !important;
            font-weight: 500 !important;
            box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05) !important;
            transition: all 0.2s ease !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
        }
        [data-testid="stFileUploader"] button:hover {
            background-color: #F8FAFC !important;
            border-color: #CBD5E1 !important;
            color: #1E293B !important;
        }
        /* Ocultar todos los elementos hijos originales para evitar duplicados y bugs de traducción */
        [data-testid="stFileUploader"] button * {
            display: none !important;
        }
        /* Mostrar el texto personalizado de forma segura */
        [data-testid="stFileUploader"] button::before {
            content: "Examinar archivos" !important;
            font-size: 0.85rem !important;
            color: inherit !important;
            display: inline-block !important;
        }
        /* Ocultar el texto duplicado de la zona de arrastre */
        [data-testid="stFileUploader"] section div {
            font-size: 0.8rem !important;
            color: #64748B !important;
        }
        </style>

        <script>
            function makeInputsReadOnly() {
                // Seleccionar todos los elementos input de la página principal (dentro del parent de Streamlit)
                var inputs = window.parent.document.querySelectorAll('input');
                inputs.forEach(function(input) {
                    // Si el input no es de tipo número (stock de seguridad) ni file uploader
                    if (input.type !== 'number' && input.type !== 'file') {
                        // Marcar como solo lectura
                        input.readOnly = true;
                        
                        // Bloquear clicks de teclado para prevenir escritura manual
                        input.onkeypress = function(e) {
                            e.preventDefault();
                        };
                        input.onkeydown = function(e) {
                            // Permitir teclas especiales de navegación como Tab, Backspace, Delete, flechas
                            var allowedKeys = ["Tab", "ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Backspace", "Delete"];
                            if (allowedKeys.indexOf(e.key) === -1 && e.key.length === 1 && !e.ctrlKey && !e.metaKey) {
                                e.preventDefault();
                            }
                        };
                    }
                });
            }
            // Ejecutar periódicamente cada 400ms para mantener el estado ante cambios reactivos del DOM de React
            setInterval(makeInputsReadOnly, 400);
        </script>
        """
    ),
    unsafe_allow_html=True,
)

# =========================================================
# CONFIGURACIÓN POR DEFECTO (dataset de demo — Kaggle)
# =========================================================
CONFIG_DEFAULT = dict(
    ruta_datos="data/train.csv",
    columna_fecha="date",
    columna_producto="item",
    columna_tienda="store",
    columna_ventas="sales",
    tienda_a_usar=1,
    productos_a_usar=[15, 28, 22, 8, 12, 24, 9, 20, 3, 5],
    porcentaje_test=0.05,
    lags=[7, 14, 30],
    ventanas_rolling=[7, 30],
    ruta_stock="data/stock_actual.csv",
    columna_stock_producto="codigo_producto",
    columna_stock_cantidad="stock_actual",
    columna_stock_fecha="fecha_corte",
)

NOMBRES_PRODUCTOS = {
    15: "Sofá Modular de Cuero", 28: "Comedor 6 Puestos Roble", 22: "Silla Ejecutiva Ergonómica",
    8: "Mesa de Centro Rústica", 12: "Estantería Industrial Acero", 24: "Escritorio Home Office",
    9: "Cama Queen Tapizada", 20: "Cómoda 5 Cajones", 3: "Butaca Reclinable", 5: "Lámpara de Pie LED",
}

PRODUCT_SKU = {
    15: "MOB-SML-001", 28: "MOB-CDR-002", 22: "MOB-SJE-003",
    8: "MOB-MCR-004", 12: "MOB-EIA-005", 24: "MOB-EHO-006",
    9: "MOB-CQT-007", 20: "MOB-C5C-008", 3: "MOB-BR-009",
    5: "MOB-LPL-010",
}

PRODUCT_CATEGORY = {
    15: "Sala", 8: "Sala", 3: "Sala",
    28: "Comedor",
    22: "Oficina", 12: "Oficina", 24: "Oficina",
    9: "Dormitorio", 20: "Dormitorio",
    5: "Iluminación"
}


# =========================================================
# PIPELINE: carga, limpieza, features, entrenamiento
# (misma lógica validada en el notebook, empaquetada en funciones)
# =========================================================
@st.cache_data(show_spinner=False)
def cargar_y_preparar_datos(ruta_datos, columna_fecha, columna_producto, columna_tienda,
                             columna_ventas, tienda_a_usar, productos_a_usar):
    df_raw = pd.read_csv(ruta_datos, parse_dates=[columna_fecha])
    renombres = {columna_fecha: "date", columna_producto: "item", columna_ventas: "sales"}
    if columna_tienda is not None and columna_tienda in df_raw.columns:
        renombres[columna_tienda] = "store"
    df_raw = df_raw.rename(columns=renombres)

    df = df_raw.copy()
    if "store" in df.columns and tienda_a_usar is not None:
        df = df[df["store"] == tienda_a_usar]
    if productos_a_usar is not None:
        df = df[df["item"].isin(productos_a_usar)]

    df = df.sort_values(["item", "date"]).reset_index(drop=True)
    return df


RUTA_MODELO_ONNX = "modelo/modelo.onnx"
RUTA_METADATA = "modelo/metadata.json"
RUTA_HISTORICO = "modelo/historico_features.parquet"


@st.cache_resource(show_spinner=False)
def cargar_modelo_preentrenado():
    """
    Carga el modelo YA ENTRENADO desde el archivo modelo.onnx (el mismo que se
    entrega como evidencia del entrenamiento), junto con la metadata necesaria
    para construir las variables de predicción (mapa de rotación, columnas usadas,
    lags, ventanas móviles) y el histórico ya procesado con sus variables de
    calendario/lags/rolling. No se reentrena nada en esta ruta.
    """
    import json

    logging.info(f"Iniciando carga de metadatos desde {RUTA_METADATA}...")
    with open(RUTA_METADATA) as f:
        metadata = json.load(f)

    logging.info(f"Cargando modelo de red predictora ONNX desde {RUTA_MODELO_ONNX}...")
    modelo = ModeloONNX(RUTA_MODELO_ONNX)
    
    logging.info(f"Cargando dataset histórico desde {RUTA_HISTORICO}...")
    df_feat = pd.read_parquet(RUTA_HISTORICO)
    
    mapa_rotacion = {int(k): v for k, v in metadata["mapa_rotacion"].items()}
    logging.info(f"Modelo cargado correctamente. Total de productos activos en catálogo: {len(metadata['productos_a_usar'])}. Error promedio (MAE): {metadata['mae']:.2f}")

    return {
        "modelo": modelo, "df_feat": df_feat, "columnas_features": metadata["columnas_features"],
        "mapa_rotacion": mapa_rotacion, "lags": metadata["lags"], "ventanas_rolling": metadata["ventanas_rolling"],
        "mae": metadata["mae"], "promedio_venta": metadata["promedio_venta"],
        "productos_a_usar": metadata["productos_a_usar"],
    }


@st.cache_resource(show_spinner=False)
def entrenar_modelo(df, porcentaje_test, lags, ventanas_rolling):
    df_feat = df.copy()

    # --- Variables de calendario ---
    df_feat["dia_semana_num"] = df_feat["date"].dt.dayofweek
    df_feat["dia_mes"] = df_feat["date"].dt.day
    df_feat["mes"] = df_feat["date"].dt.month
    df_feat["trimestre"] = df_feat["date"].dt.quarter
    df_feat["anio"] = df_feat["date"].dt.year
    df_feat["es_fin_de_semana"] = df_feat["dia_semana_num"].isin([5, 6]).astype(int)

    # --- Categoría de rotación (terciles automáticos) ---
    promedio_producto = df_feat.groupby("item")["sales"].mean()
    categorias = pd.qcut(promedio_producto, q=3, labels=["baja", "media", "alta"])
    mapa_rotacion = categorias.to_dict()
    df_feat["categoria_rotacion"] = df_feat["item"].map(mapa_rotacion)
    df_feat = pd.get_dummies(df_feat, columns=["categoria_rotacion"], prefix="rotacion")
    columnas_rotacion = [c for c in df_feat.columns if c.startswith("rotacion_")]

    # --- Lags y ventanas móviles ---
    df_feat = df_feat.sort_values(["item", "date"]).reset_index(drop=True)
    for lag in lags:
        df_feat[f"lag_{lag}"] = df_feat.groupby("item")["sales"].shift(lag)
    for w in ventanas_rolling:
        df_feat[f"media_movil_{w}"] = df_feat.groupby("item")["sales"].transform(lambda x: x.rolling(w).mean())
    df_feat[f"std_movil_{ventanas_rolling[0]}"] = df_feat.groupby("item")["sales"].transform(
        lambda x: x.rolling(ventanas_rolling[0]).std()
    )
    df_feat = df_feat.dropna().reset_index(drop=True)

    columnas_lag = [f"lag_{l}" for l in lags]
    columnas_roll = [f"media_movil_{w}" for w in ventanas_rolling] + [f"std_movil_{ventanas_rolling[0]}"]
    columnas_features = (
        ["dia_semana_num", "dia_mes", "mes", "trimestre", "anio", "es_fin_de_semana"]
        + columnas_rotacion + columnas_lag + columnas_roll
    )

    # --- Split temporal por porcentaje ---
    rango_total_dias = (df_feat["date"].max() - df_feat["date"].min()).days
    dias_test = int(rango_total_dias * porcentaje_test)
    fecha_corte_split = df_feat["date"].max() - pd.Timedelta(days=dias_test)

    train = df_feat[df_feat["date"] <= fecha_corte_split]
    test = df_feat[df_feat["date"] > fecha_corte_split]

    X_train, y_train = train[columnas_features], train["sales"]
    X_test, y_test = test[columnas_features], test["sales"]

    modelo = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                           subsample=0.8, colsample_bytree=0.8, random_state=42)
    modelo.fit(X_train, y_train)

    mae = mean_absolute_error(y_test, modelo.predict(X_test)) if len(X_test) > 0 else np.nan

    return {
        "modelo": modelo, "df_feat": df_feat, "columnas_features": columnas_features,
        "mapa_rotacion": mapa_rotacion, "lags": lags, "ventanas_rolling": ventanas_rolling,
        "mae": mae, "promedio_venta": df_feat["sales"].mean(),
    }


@st.cache_data(show_spinner=False)
def cargar_stock(ruta_stock, columna_producto, columna_cantidad, columna_fecha):
    stock_df = pd.read_csv(ruta_stock, parse_dates=[columna_fecha])
    stock_df = stock_df.rename(columns={
        columna_producto: "item", columna_cantidad: "stock_actual", columna_fecha: "fecha_corte"
    })
    stock_dict = stock_df.set_index("item")["stock_actual"].to_dict()
    fecha_corte_stock = stock_df["fecha_corte"].max()
    return stock_dict, fecha_corte_stock, stock_df


# =========================================================
# MOTOR DE PREDICCIÓN (idéntico al validado en el notebook)
# =========================================================
def preparar_features(item_id, fecha_objetivo, df_historico, modelo_info):
    fecha_objetivo = pd.to_datetime(fecha_objetivo)
    lags, ventanas = modelo_info["lags"], modelo_info["ventanas_rolling"]

    historial = df_historico[
        (df_historico["item"] == item_id) & (df_historico["date"] < fecha_objetivo)
    ].sort_values("date")

    if len(historial) < max(lags):
        raise ValueError(f"No hay suficiente historial para el producto {item_id} antes de {fecha_objetivo.date()}")

    ventas = historial["sales"].values
    features = {
        "dia_semana_num": fecha_objetivo.dayofweek, "dia_mes": fecha_objetivo.day,
        "mes": fecha_objetivo.month, "trimestre": fecha_objetivo.quarter, "anio": fecha_objetivo.year,
        "es_fin_de_semana": int(fecha_objetivo.dayofweek in [5, 6]),
    }
    for lag in lags:
        features[f"lag_{lag}"] = ventas[-lag]
    for w in ventanas:
        features[f"media_movil_{w}"] = ventas[-w:].mean()
    features[f"std_movil_{ventanas[0]}"] = ventas[-ventanas[0]:].std()

    categoria = modelo_info["mapa_rotacion"][item_id]
    for cat in ["alta", "media", "baja"]:
        features[f"rotacion_{cat}"] = int(categoria == cat)

    return pd.DataFrame([features])[modelo_info["columnas_features"]]


def predecir_recursivo(item_id, fecha_inicio, fecha_fin, df_historico, modelo_info):
    logging.info(f"[PROCESO] Iniciando prediccion recursiva para Producto {item_id} desde {fecha_inicio.date()} hasta {fecha_fin.date()}...")
    fecha_inicio_dt = pd.to_datetime(fecha_inicio)
    historico_ext = df_historico[
        (df_historico["item"] == item_id) & (df_historico["date"] < fecha_inicio_dt)
    ].copy()
    fechas = pd.date_range(fecha_inicio, fecha_fin, freq="D")

    preds = []
    for fecha in fechas:
        X_pred = preparar_features(item_id, fecha, historico_ext, modelo_info)
        pred = modelo_info["modelo"].predict(X_pred)[0]
        preds.append({"date": fecha, "item": item_id, "sales": pred})
        nueva = pd.DataFrame([{"date": fecha, "item": item_id, "sales": pred}])
        historico_ext = pd.concat([historico_ext, nueva], ignore_index=True)

    logging.info(f"[PROCESO] Prediccion completada para Producto {item_id}. Dias proyectados: {len(preds)}")
    return pd.DataFrame(preds)


def cuanto_pedir(item_id, fecha_objetivo, modelo_info, stock_actual=0, stock_seguridad=0,
                  fecha_corte_stock=None):
    fecha_objetivo = pd.to_datetime(fecha_objetivo)
    df_historico = modelo_info["df_feat"]
    historial_item = df_historico[df_historico["item"] == item_id]
    fecha_max_real = historial_item["date"].max()
    fecha_inicio_conteo = fecha_corte_stock + pd.Timedelta(days=1)

    consumo_real = 0
    if fecha_max_real >= fecha_inicio_conteo:
        tramo_real = historial_item[
            (historial_item["date"] >= fecha_inicio_conteo)
            & (historial_item["date"] <= min(fecha_max_real, fecha_objetivo))
        ]
        consumo_real = tramo_real["sales"].sum()
        fecha_inicio_prediccion = fecha_max_real + pd.Timedelta(days=1)
    else:
        fecha_inicio_prediccion = fecha_inicio_conteo

    demanda_predicha = 0
    detalle_diario = pd.DataFrame()
    if fecha_inicio_prediccion <= fecha_objetivo:
        detalle_diario = predecir_recursivo(item_id, fecha_inicio_prediccion, fecha_objetivo, df_historico, modelo_info)
        demanda_predicha = detalle_diario["sales"].sum()

    demanda_total = consumo_real + demanda_predicha
    cantidad_a_pedir = max(0, round(demanda_total + stock_seguridad - stock_actual))

    logging.info(f"[LOGÍSTICA] Producto {item_id} | Stock Actual: {stock_actual} | Stock Seguridad: {stock_seguridad} | Demanda Total Proyectada: {round(demanda_total)} | Sugerencia de Compra: {cantidad_a_pedir}")

    return {
        "demanda_total": round(demanda_total), "consumo_real": round(consumo_real),
        "demanda_predicha": round(demanda_predicha), "cantidad_a_pedir": cantidad_a_pedir,
        "detalle_diario": detalle_diario,
    }


# =========================================================
# INTERFAZ
# =========================================================
st.markdown(
    textwrap.dedent(
        """
        <div style="padding: 0rem 0 1.5rem 0;">
            <h1 style="margin: 0; font-size: 2.4rem; font-weight: 700; color: #1E293B;">Panel de Inteligencia Predictiva</h1>
            <p style="color: #64748B; font-size: 1.05rem; margin-top: 0.3rem;">
            Consulta el stock, predice la demanda y genera órdenes de compra automáticas con IA.
            </p>
        </div>
        """
    ),
    unsafe_allow_html=True,
)

# Inyectar logs iniciales en la consola Chrome/DevTools del navegador del usuario
st.components.v1.html(
    """
    <script>
        console.log('%c[SISTEMA] Panel de Inteligencia Predictiva Inicializado.', 'color: #16A34A; font-weight: bold; font-size: 1.1em;');
        console.log('%c[SISTEMA] Modelo ONNX cargado exitosamente. Listo para operaciones.', 'color: #475569; font-weight: 500;');
    </script>
    """,
    height=0
)

# --- Configuración de datos por defecto (sin panel de configuración) ---
fuente = "Usar datos de demostración (Kaggle)"
ventas_file, stock_file = None, None

# --- Carga de datos y entrenamiento (con caché para no reentrenar en cada clic) ---
with st.spinner("Preparando datos y entrenando el modelo..."):
    if fuente == "Subir ambos" and ventas_file and stock_file:
        df_ventas_raw = pd.read_csv(ventas_file, parse_dates=[0])
        df_ventas_raw.columns = ["date", "item", "sales"] if len(df_ventas_raw.columns) == 3 else df_ventas_raw.columns
        df = df_ventas_raw.sort_values(["item", "date"]).reset_index(drop=True)
        modelo_info = entrenar_modelo(df, CONFIG_DEFAULT["porcentaje_test"], CONFIG_DEFAULT["lags"], CONFIG_DEFAULT["ventanas_rolling"])
        modelo_recien_entrenado = True

        stock_df_raw = pd.read_csv(stock_file)
        stock_df_raw.columns = ["item", "stock_actual", "fecha_corte"] if len(stock_df_raw.columns) == 3 else stock_df_raw.columns
        stock_df_raw["fecha_corte"] = pd.to_datetime(stock_df_raw["fecha_corte"])
        stock_dict = stock_df_raw.set_index("item")["stock_actual"].to_dict()
        fecha_corte_stock = stock_df_raw["fecha_corte"].max()
        productos_activos = sorted(modelo_info["df_feat"]["item"].unique().tolist())
        nombres_activos = {p: f"Producto {p}" for p in productos_activos}
    else:
        modelo_info = cargar_modelo_preentrenado()
        modelo_recien_entrenado = False
        stock_dict, fecha_corte_stock, _ = cargar_stock(
            CONFIG_DEFAULT["ruta_stock"], CONFIG_DEFAULT["columna_stock_producto"],
            CONFIG_DEFAULT["columna_stock_cantidad"], CONFIG_DEFAULT["columna_stock_fecha"],
        )
        productos_activos = modelo_info["productos_a_usar"]
        nombres_activos = NOMBRES_PRODUCTOS



tab1, tab2 = st.tabs(["Consulta Específica", "Consulta General (todo el catálogo)"])

# =========================================================
# TAB 1 — Consulta específica (un producto)
# =========================================================
with tab1:
    with st.container(border=True):
        col_title, col_toggle = st.columns([3, 2])
        with col_title:
            st.markdown(
                textwrap.dedent(
                    """
                    <h3 style="margin: 0; font-size: 1.25rem; font-weight: 600; color: #1E293B; display: flex; align-items: center; gap: 8px;">
                        <svg style="width: 20px; height: 20px; color: #2563EB;" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
                            <path stroke-linecap="round" stroke-linejoin="round" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                        </svg>
                        Consulta Predictiva Específica
                    </h3>
                    """
                ),
                unsafe_allow_html=True
            )
        with col_toggle:
            modo_fecha = st.segmented_control(
                "Modo de fecha",
                ["Fecha Única", "Rango de Fechas"],
                default="Fecha Única",
                key="modo_esp",
                label_visibility="collapsed"
            )

        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
        
        col_cat, col_prod = st.columns(2)
        with col_cat:
            unique_categories = sorted(list(set(PRODUCT_CATEGORY.values())))
            categorias_opciones = ["Todas las Categorías"] + unique_categories
            categoria_sel = st.selectbox("Categoría", categorias_opciones, key="cat_esp")

        with col_prod:
            if categoria_sel == "Todas las Categorías":
                productos_filtrados = productos_activos
            else:
                productos_filtrados = [p for p in productos_activos if PRODUCT_CATEGORY.get(p, "General") == categoria_sel]

            producto_sel = st.selectbox(
                "Artículo de inventario",
                productos_filtrados,
                format_func=lambda x: f"{nombres_activos.get(x, f'Producto {x}')} (# {x})",
                key="prod_esp",
            )
        
        # Subtexto con SKU y Stock Actual
        sku_esp = PRODUCT_SKU.get(producto_sel, f"SKU-{producto_sel}")
        stock_prod = stock_dict.get(producto_sel, 0)
        st.markdown(
            textwrap.dedent(
                f"""
                <div style='font-size: 0.85rem; color: #64748B; margin-top: -12px; margin-bottom: 12px;'>SKU: {sku_esp} &bull; Stock: {stock_prod} uds.</div>
                """
            ),
            unsafe_allow_html=True
        )

        fecha_min_permitida = fecha_corte_stock + pd.Timedelta(days=1)
        label_fecha_desde = "Fecha Objetivo a Predecir" if modo_fecha == "Fecha Única" else "Fecha Inicio (Desde)"
        c1, c2 = st.columns(2)
        with c1:
            fecha_desde = st.date_input(label_fecha_desde, value=fecha_min_permitida, min_value=fecha_min_permitida, key="desde_esp")
        with c2:
            if modo_fecha == "Rango de Fechas":
                fecha_hasta = st.date_input("Fecha Fin (Hasta)", value=fecha_desde + pd.Timedelta(days=30),
                                             min_value=fecha_desde, key="hasta_esp")
            else:
                fecha_hasta = fecha_desde
                st.markdown("<div style='height: 0px; margin-top: 28px;'></div>", unsafe_allow_html=True)
                st.caption("Predicción para un solo día de corte objetivo.")

        stock_seguridad_esp = st.number_input("Stock de seguridad (opcional)", min_value=0, value=0, key="seg_esp")

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        btn_esp_clicked = st.button("Consultar IA Predictiva Específica", type="primary", use_container_width=True, key="btn_esp")

    if btn_esp_clicked:
        try:
            resultado = cuanto_pedir(
                producto_sel, pd.Timestamp(fecha_hasta), modelo_info,
                stock_actual=stock_prod, stock_seguridad=stock_seguridad_esp,
                fecha_corte_stock=fecha_corte_stock,
            )

            # Inyectar log en la consola de Chrome/DevTools del navegador
            st.components.v1.html(
                f"""
                <script>
                    console.log('%c[LOGÍSTICA] Producto {producto_sel} | Stock Actual: {stock_prod} | Stock Seguridad: {stock_seguridad_esp} | Demanda Proyectada: {resultado["demanda_total"]} | Sugerencia Compra: {resultado["cantidad_a_pedir"]}', 'color: #2563EB; font-weight: bold;');
                </script>
                """,
                height=0
            )

            p_name = nombres_activos.get(producto_sel, f"Producto {producto_sel}")
            fecha_desde_str = fecha_desde.strftime('%d de %B de %Y')
            fecha_hasta_str = fecha_hasta.strftime('%d de %B de %Y')
            
            st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
            
            # Badge y Título
            st.markdown(
                textwrap.dedent(
                    f"""
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; margin-bottom: 16px;">
                        <div>
                            <h2 style="margin: 0; font-size: 1.8rem; font-weight: 700; color: #1E293B;">{p_name}</h2>
                            <p style="color: #64748B; font-size: 0.95rem; margin-top: 4px; margin-bottom: 0;">Análisis predictivo desde el {fecha_desde_str} hasta el {fecha_hasta_str}</p>
                        </div>
                        <div style="background: #DEF7EC; color: #03543F; font-size: 0.8rem; font-weight: 600; padding: 6px 12px; border-radius: 9999px; display: flex; align-items: center; gap: 4px;">
                            <svg style="width: 14px; height: 14px;" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2.5">
                                <path stroke-linecap="round" stroke-linejoin="round" d="M5 13l4 4L19 7" />
                            </svg>
                            Análisis completado
                        </div>
                    </div>
                    """
                ),
                unsafe_allow_html=True
            )

            # KPIs en Tarjetas Custom HTML
            demand_proj = resultado["demanda_total"]
            order_qty = resultado["cantidad_a_pedir"]
            coverage_pct = int(round((stock_prod / demand_proj) * 100)) if demand_proj > 0 else 100
            bar_width = min(100, coverage_pct)

            st.markdown(
                textwrap.dedent(
                    f"""<div style="display: flex; gap: 16px; margin-bottom: 24px; flex-wrap: wrap; width: 100%;">
                        <!-- CARD 1 -->
                        <div style="flex: 1; min-width: 250px; background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); display: flex; flex-direction: column; justify-content: space-between;">
                            <div>
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                                    <div style="background: #EFF6FF; padding: 8px; border-radius: 8px; display: flex; align-items: center; justify-content: center; width: 36px; height: 36px;">
                                        <svg style="width: 20px; height: 20px; color: #2563EB;" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
                                            <path stroke-linecap="round" stroke-linejoin="round" d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" />
                                        </svg>
                                    </div>
                                    <div style="display: flex; align-items: center; gap: 4px; color: #64748B; font-size: 0.85rem;">
                                        <svg style="width: 14px; height: 14px;" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
                                            <path stroke-linecap="round" stroke-linejoin="round" d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4" />
                                        </svg>
                                        PostgreSQL
                                    </div>
                                </div>
                                <div style="font-size: 1.8rem; font-weight: 700; color: #1E293B; line-height: 1.2;">
                                    {stock_prod} <span style="font-size: 1rem; font-weight: 500; color: #64748B;">uds.</span>
                                </div>
                                <div style="color: #64748B; font-size: 0.9rem; margin-top: 4px; margin-bottom: 8px;">Stock Actual</div>
                            </div>
                            <div style="display: flex; align-items: center; gap: 8px; font-size: 0.8rem; color: #64748B; margin-top: 12px; width: 100%;">
                                <div style="flex: 1; background: #E2E8F0; height: 6px; border-radius: 3px; overflow: hidden;">
                                    <div style="background: #3B82F6; width: {bar_width}%; height: 100%;"></div>
                                </div>
                                <span>{coverage_pct}% de cobertura</span>
                            </div>
                        </div>
                        <!-- CARD 2 -->
                        <div style="flex: 1; min-width: 250px; background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); display: flex; flex-direction: column; justify-content: space-between;">
                            <div>
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                                    <div style="background: #F3E8FF; padding: 8px; border-radius: 8px; display: flex; align-items: center; justify-content: center; width: 36px; height: 36px;">
                                        <svg style="width: 20px; height: 20px; color: #8B5CF6;" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
                                            <path stroke-linecap="round" stroke-linejoin="round" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                                        </svg>
                                    </div>
                                    <div style="display: flex; align-items: center; gap: 4px; color: #64748B; font-size: 0.85rem;">
                                        <svg style="width: 14px; height: 14px;" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
                                            <path stroke-linecap="round" stroke-linejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z" />
                                        </svg>
                                        FastAPI
                                    </div>
                                </div>
                                <div style="font-size: 1.8rem; font-weight: 700; color: #1E293B; line-height: 1.2;">
                                    {demand_proj} <span style="font-size: 1rem; font-weight: 500; color: #64748B;">uds.</span>
                                </div>
                                <div style="color: #64748B; font-size: 0.9rem; margin-top: 4px; margin-bottom: 8px;">Predicción de Ventas</div>
                            </div>
                            <div style="display: flex; align-items: center; gap: 4px; font-size: 0.8rem; color: #8B5CF6; margin-top: 12px; font-weight: 600;">
                                <svg style="width: 14px; height: 14px;" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
                                    <path stroke-linecap="round" stroke-linejoin="round" d="M7 12l3-3 3 3 4-4M8 21h8a2 2 0 002-2V5a2 2 0 00-2-2H8a2 2 0 00-2 2v14a2 2 0 002 2z" />
                                </svg>
                                Demanda proyectada por IA
                            </div>
                        </div>
                        <!-- CARD 3 -->
                        <div style="flex: 1; min-width: 250px; background: linear-gradient(135deg, #10B981 0%, #059669 100%); border: none; border-radius: 12px; padding: 20px; box-shadow: 0 4px 12px rgba(16,185,129,0.25); display: flex; flex-direction: column; justify-content: space-between;">
                            <div>
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                                    <div style="background: rgba(255,255,255,0.2); padding: 8px; border-radius: 8px; display: flex; align-items: center; justify-content: center; width: 36px; height: 36px;">
                                        <svg style="width: 20px; height: 20px; color: #FFFFFF;" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
                                            <path stroke-linecap="round" stroke-linejoin="round" d="M3 3h2l.4 2M7 13h10l4-8H5.4M7 13L5.4 5M7 13l-2.293 2.293c-.63.63-.184 1.707.707 1.707H17m0 0a2 2 0 100 4 2 2 0 000-4zm-8 2a2 2 0 11-4 0 2 2 0 014 0z" />
                                        </svg>
                                    </div>
                                    <div style="display: flex; align-items: center; gap: 4px; color: rgba(255,255,255,0.9); font-size: 0.85rem;">
                                        <svg style="width: 14px; height: 14px;" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
                                            <path stroke-linecap="round" stroke-linejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                                        </svg>
                                        Calculado
                                    </div>
                                </div>
                                <div style="font-size: 1.8rem; font-weight: 700; color: #FFFFFF; line-height: 1.2;">
                                    {order_qty} <span style="font-size: 1rem; font-weight: 500; color: rgba(255,255,255,0.85);">uds.</span>
                                </div>
                                <div style="color: rgba(255,255,255,0.9); font-size: 0.9rem; margin-top: 4px; margin-bottom: 8px;">Orden de Compra</div>
                            </div>
                            <div style="display: flex; align-items: center; gap: 4px; font-size: 0.8rem; color: #FFFFFF; margin-top: 12px; font-weight: 600;">
                                <svg style="width: 14px; height: 14px;" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
                                    <path stroke-linecap="round" stroke-linejoin="round" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
                                </svg>
                                Cubre demanda + {stock_seguridad_esp} uds. de seguridad
                            </div>
                        </div>
                    </div>"""
                ),
                unsafe_allow_html=True
            )

            # --- 1. Gráfico Histórico + Predicción en Light Theme (Pestaña 1) ---
            df_historico = modelo_info["df_feat"]
            historial_item = df_historico[df_historico["item"] == producto_sel].sort_values("date")
            historial_item_reciente = historial_item[historial_item["date"] >= (fecha_corte_stock - pd.Timedelta(days=90))]
            
            fig1 = go.Figure()
            # Ventas históricas reales
            fig1.add_trace(go.Scatter(
                x=historial_item_reciente["date"],
                y=historial_item_reciente["sales"],
                name="Ventas Históricas (Últimos 90 días)",
                mode="lines+markers",
                line=dict(color="#3B82F6", width=2),
                marker=dict(size=4)
            ))
            # Ventas futuras proyectadas (IA)
            if not resultado["detalle_diario"].empty:
                ultimo_real = historial_item_reciente.iloc[-1] if not historial_item_reciente.empty else None
                df_pred = resultado["detalle_diario"].sort_values("date")
                x_pred = df_pred["date"]
                y_pred = df_pred["sales"]
                if ultimo_real is not None:
                    x_pred = pd.concat([pd.Series([ultimo_real["date"]]), x_pred], ignore_index=True)
                    y_pred = pd.concat([pd.Series([ultimo_real["sales"]]), y_pred], ignore_index=True)
                fig1.add_trace(go.Scatter(
                    x=x_pred,
                    y=y_pred,
                    name="Proyección IA (XGBoost)",
                    mode="lines+markers",
                    line=dict(color="#8B5CF6", width=2, dash="dash"),
                    marker=dict(size=4)
                ))
            fig1.update_layout(
                title=f"Evolución y Predicción de Demanda - {p_name}",
                height=340,
                margin=dict(l=10, r=10, t=45, b=40),
                legend=dict(orientation="h", y=1.1, font=dict(color="#475569")),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter, sans-serif", color="#1E293B"),
                xaxis=dict(gridcolor="#F1F5F9", tickfont=dict(color="#475569")),
                yaxis=dict(gridcolor="#F1F5F9", tickfont=dict(color="#475569"))
            )
            st.plotly_chart(fig1, use_container_width=True)

            # --- 2. Desglose y Recomendación organizados en dos columnas ---
            max_val = max(1, demand_proj, stock_prod, stock_seguridad_esp, order_qty)
            width_demand = (demand_proj / max_val) * 100
            width_stock = (stock_prod / max_val) * 100
            width_safety = (stock_seguridad_esp / max_val) * 100
            width_order = (order_qty / max_val) * 100

            col_desglose, col_recomendacion = st.columns([1, 1])
            with col_desglose:
                st.markdown(
                    textwrap.dedent(
                        f"""<div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); height: 100%;">
                            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 20px;">
                                <svg style="width: 18px; height: 18px; color: #64748B;" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
                                    <path stroke-linecap="round" stroke-linejoin="round" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 002 2h2a2 2 0 002-2z" />
                                </svg>
                                <h4 style="margin: 0; font-size: 1.1rem; font-weight: 600; color: #1E293B;">Desglose del cálculo</h4>
                            </div>
                            <!-- Row 1 -->
                            <div style="display: flex; align-items: center; margin-bottom: 14px; flex-wrap: wrap;">
                                <div style="width: 180px; font-size: 0.9rem; color: #475569;">Demanda proyectada</div>
                                <div style="flex: 1; min-width: 100px; background: #F1F5F9; height: 10px; border-radius: 5px; overflow: hidden; margin-right: 16px;">
                                    <div style="background: #8B5CF6; width: {width_demand}%; height: 100%; border-radius: 5px;"></div>
                                </div>
                                <div style="width: 40px; text-align: right; font-weight: 600; color: #1E293B; font-size: 0.9rem;">{demand_proj}</div>
                            </div>
                            <!-- Row 2 -->
                            <div style="display: flex; align-items: center; margin-bottom: 14px; flex-wrap: wrap;">
                                <div style="width: 180px; font-size: 0.9rem; color: #475569;">Stock actual disponible</div>
                                <div style="flex: 1; min-width: 100px; background: #F1F5F9; height: 10px; border-radius: 5px; overflow: hidden; margin-right: 16px;">
                                    <div style="background: #3B82F6; width: {width_stock}%; height: 100%; border-radius: 5px;"></div>
                                </div>
                                <div style="width: 40px; text-align: right; font-weight: 600; color: #1E293B; font-size: 0.9rem;">{stock_prod}</div>
                            </div>
                            <!-- Row 3 -->
                            <div style="display: flex; align-items: center; margin-bottom: 14px; flex-wrap: wrap;">
                                <div style="width: 180px; font-size: 0.9rem; color: #475569;">Stock de seguridad</div>
                                <div style="flex: 1; min-width: 100px; background: #F1F5F9; height: 10px; border-radius: 5px; overflow: hidden; margin-right: 16px;">
                                    <div style="background: #E2E8F0; width: {width_safety}%; height: 100%; border-radius: 5px;"></div>
                                </div>
                                <div style="width: 40px; text-align: right; font-weight: 600; color: #1E293B; font-size: 0.9rem;">{stock_seguridad_esp}</div>
                            </div>
                            <!-- Row 4 -->
                            <div style="display: flex; align-items: center; flex-wrap: wrap;">
                                <div style="width: 180px; font-size: 0.9rem; color: #475569; font-weight: 600;">Sugerencia compra</div>
                                <div style="flex: 1; min-width: 100px; background: #F1F5F9; height: 10px; border-radius: 5px; overflow: hidden; margin-right: 16px;">
                                    <div style="background: #10B981; width: {width_order}%; height: 100%; border-radius: 5px;"></div>
                                </div>
                                <div style="width: 40px; text-align: right; font-weight: 600; color: #10B981; font-size: 0.9rem;">{order_qty}</div>
                            </div>
                        </div>"""
                    ),
                    unsafe_allow_html=True
                )
            
            with col_recomendacion:
                st.markdown(
                    textwrap.dedent(
                        f"""<div style="display: flex; gap: 12px; align-items: flex-start; background: #FFFBEB; border: 1px solid #FDE68A; border-radius: 12px; padding: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); height: 100%;">
                            <div style="background: #FEF3C7; padding: 6px; border-radius: 6px; color: #D97706; display: flex; align-items: center; justify-content: center; width: 28px; height: 28px; flex-shrink: 0;">
                                <svg style="width: 16px; height: 16px;" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
                                    <path stroke-linecap="round" stroke-linejoin="round" d="M3 3h2l.4 2M7 13h10l4-8H5.4M7 13L5.4 5M7 13l-2.293 2.293c-.63.63-.184 1.707.707 1.707H17m0 0a2 2 0 100 4 2 2 0 000-4zm-8 2a2 2 0 11-4 0 2 2 0 014 0z" />
                                </svg>
                            </div>
                            <div style="font-size: 0.9rem; color: #92400E; line-height: 1.5; flex: 1;">
                                <strong style="color: #78350F;">Recomendación de compra</strong><br>
                                Emitir orden a proveedor por <strong style="color: #78350F;">{order_qty} unidades</strong> de {p_name}.<br>
                                <span style="font-size: 0.8rem; color: #B45309;">Fórmula: ({demand_proj} demanda &minus; {stock_prod} stock) + {stock_seguridad_esp} seguridad.</span>
                            </div>
                        </div>"""
                    ),
                    unsafe_allow_html=True
                )
                if resultado["demanda_predicha"] > 0 and resultado["consumo_real"] > 0:
                    st.caption(f"De la demanda, {resultado['consumo_real']} uds. son ventas reales y {resultado['demanda_predicha']} uds. son proyecciones.")
                st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
                csv_diario = resultado["detalle_diario"].to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Descargar Reporte de Proyección Diaria (CSV)",
                    data=csv_diario,
                    file_name=f"proyeccion_diaria_{producto_sel}.csv",
                    mime="text/csv",
                    key=f"btn_download_diario_{producto_sel}",
                    use_container_width=True
                )

        except ValueError as e:
            st.error(f"No se pudo calcular: {e}")

# =========================================================
# TAB 2 — Consulta general (todo el catálogo)
# =========================================================
with tab2:
    with st.container(border=True):
        col_title_g, col_toggle_g = st.columns([3, 2])
        with col_title_g:
            st.markdown(
                textwrap.dedent(
                    """
                    <h3 style="margin: 0; font-size: 1.25rem; font-weight: 600; color: #1E293B; display: flex; align-items: center; gap: 8px;">
                        <svg style="width: 20px; height: 20px; color: #8B5CF6;" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
                            <path stroke-linecap="round" stroke-linejoin="round" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                        </svg>
                        Consulta Predictiva General
                    </h3>
                    """
                ),
                unsafe_allow_html=True
            )
        with col_toggle_g:
            modo_fecha_g = st.segmented_control(
                "Modo de fecha general",
                ["Fecha Única", "Rango de Fechas"],
                default="Fecha Única",
                key="modo_gen",
                label_visibility="collapsed"
            )

        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

        categoria_g = st.selectbox("Categoría a Consultar", categorias_opciones, key="cat_gen")

        if categoria_g == "Todas las Categorías":
            productos_analisis = productos_activos
        else:
            productos_analisis = [p for p in productos_activos if PRODUCT_CATEGORY.get(p, "General") == categoria_g]

        st.markdown(
            textwrap.dedent(
                f"""
                <div style='font-size: 0.85rem; color: #64748B; margin-top: -12px; margin-bottom: 12px;'>Se analizará la demanda global de los {len(productos_analisis)} artículos seleccionados.</div>
                """
            ),
            unsafe_allow_html=True
        )

        fecha_min_permitida_g = fecha_corte_stock + pd.Timedelta(days=1)
        label_fecha_desde_g = "Fecha Objetivo a Predecir" if modo_fecha_g == "Fecha Única" else "Fecha Inicio (Desde)"
        c1_g, c2_g = st.columns(2)
        with c1_g:
            fecha_desde_g = st.date_input(label_fecha_desde_g, value=fecha_min_permitida_g, min_value=fecha_min_permitida_g, key="desde_gen")
        with c2_g:
            if modo_fecha_g == "Rango de Fechas":
                fecha_hasta_g = st.date_input("Fecha Fin (Hasta)", value=fecha_desde_g + pd.Timedelta(days=30),
                                               min_value=fecha_desde_g, key="hasta_gen")
            else:
                fecha_hasta_g = fecha_desde_g
                st.markdown("<div style='height: 0px; margin-top: 28px;'></div>", unsafe_allow_html=True)
                st.caption("Predicción para un solo día de corte objetivo.")

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        
        # Envolvemos el botón en una clase CSS para hacerlo púrpura
        st.markdown("<div class='tab-general-btn'>", unsafe_allow_html=True)
        btn_gen_clicked = st.button("Consultar IA Predictiva General", type="primary", use_container_width=True, key="btn_gen")
        st.markdown("</div>", unsafe_allow_html=True)

    if btn_gen_clicked:
        st.components.v1.html(
            f"""
            <script>
                console.log('%c[SISTEMA] Iniciando analisis global de demanda para {len(productos_analisis)} productos...', 'color: #8B5CF6; font-weight: bold;');
            </script>
            """,
            height=0
        )
        filas = []
        barra = st.progress(0.0, text="Calculando predicciones por producto...")
        
        for i, item_id in enumerate(productos_analisis):
            # Obtener predicción base (con seguridad 0 de base)
            r_temp = cuanto_pedir(
                item_id, pd.Timestamp(fecha_hasta_g), modelo_info,
                stock_actual=stock_dict.get(item_id, 0), stock_seguridad=0,
                fecha_corte_stock=fecha_corte_stock,
            )
            demanda_g = r_temp["demanda_total"]
            stock_act_g = stock_dict.get(item_id, 0)
            safety_g = int(np.ceil(demanda_g * 0.15))
            order_g = max(0, demanda_g + safety_g - stock_act_g)
            
            filas.append({
                "Producto": nombres_activos.get(item_id, f"Producto {item_id}"),
                "SKU": f"#{item_id}",
                "Stock Actual": stock_act_g,
                "Demanda Proyectada": demanda_g,
                "Stock Seguridad": safety_g,
                "Sugerencia de Pedido": order_g,
            })
            barra.progress((i + 1) / len(productos_analisis))
        barra.empty()

        tabla = pd.DataFrame(filas).sort_values("Sugerencia de Pedido", ascending=False).reset_index(drop=True)
        compra_total = tabla["Sugerencia de Pedido"].sum()
        
        st.components.v1.html(
            f"""
            <script>
                console.log('%c[SISTEMA] Analisis global completado. Compra sugerida consolidada: {compra_total} unidades.', 'color: #16A34A; font-weight: bold;');
            </script>
            """,
            height=0
        )

        fecha_desde_g_str = fecha_desde_g.strftime('%d de %B de %Y')
        fecha_hasta_g_str = fecha_hasta_g.strftime('%d de %B de %Y')

        st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

        # Badge y Título
        st.markdown(
            textwrap.dedent(
                f"""
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; margin-bottom: 16px;">
                    <div>
                        <h2 style="margin: 0; font-size: 1.8rem; font-weight: 700; color: #1E293B;">Proyección General: Todos los Productos</h2>
                        <p style="color: #64748B; font-size: 0.95rem; margin-top: 4px; margin-bottom: 0;">Análisis predictivo desde el {fecha_desde_g_str} hasta el {fecha_hasta_g_str}</p>
                    </div>
                    <div style="background: #DEF7EC; color: #03543F; font-size: 0.8rem; font-weight: 600; padding: 6px 12px; border-radius: 9999px; display: flex; align-items: center; gap: 4px;">
                        <svg style="width: 14px; height: 14px;" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2.5">
                            <path stroke-linecap="round" stroke-linejoin="round" d="M5 13l4 4L19 7" />
                        </svg>
                        Análisis general completado
                    </div>
                </div>
                """
            ),
            unsafe_allow_html=True
        )

        # Cuatro tarjetas de métricas en HTML
        total_p_analizados = len(tabla)
        total_stock = tabla['Stock Actual'].sum()
        total_demand = tabla['Demanda Proyectada'].sum()
        total_order = tabla['Sugerencia de Pedido'].sum()

        st.markdown(
            textwrap.dedent(
                f"""
                <div style="display: flex; gap: 16px; margin-bottom: 24px; flex-wrap: wrap; width: 100%;">
                    <!-- CARD 1 -->
                    <div style="flex: 1; min-width: 200px; background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                        <div style="font-size: 1.8rem; font-weight: 700; color: #1E293B; line-height: 1.2;">{total_p_analizados}</div>
                        <div style="color: #64748B; font-size: 0.9rem; margin-top: 8px;">Productos Analizados</div>
                    </div>
                    <!-- CARD 2 -->
                    <div style="flex: 1; min-width: 200px; background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                        <div style="font-size: 1.8rem; font-weight: 700; color: #2563EB; line-height: 1.2;">
                            {total_stock} <span style="font-size: 1rem; font-weight: 500; color: #64748B;">uds.</span>
                        </div>
                        <div style="color: #64748B; font-size: 0.9rem; margin-top: 8px;">Stock Total Actual</div>
                    </div>
                    <!-- CARD 3 -->
                    <div style="flex: 1; min-width: 200px; background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                        <div style="font-size: 1.8rem; font-weight: 700; color: #8B5CF6; line-height: 1.2;">
                            {total_demand} <span style="font-size: 1rem; font-weight: 500; color: #64748B;">uds.</span>
                        </div>
                        <div style="color: #64748B; font-size: 0.9rem; margin-top: 8px;">Demanda Total Proyectada</div>
                    </div>
                    <!-- CARD 4 -->
                    <div style="flex: 1; min-width: 200px; background: linear-gradient(135deg, #10B981 0%, #059669 100%); border: none; border-radius: 12px; padding: 18px; box-shadow: 0 4px 12px rgba(16,185,129,0.25);">
                        <div style="font-size: 1.8rem; font-weight: 700; color: #FFFFFF; line-height: 1.2;">
                            {total_order} <span style="font-size: 1rem; font-weight: 500; color: rgba(255,255,255,0.85);">uds.</span>
                        </div>
                        <div style="color: rgba(255,255,255,0.9); font-size: 0.9rem; margin-top: 8px;">Orden de Compra Total</div>
                    </div>
                </div>
                """
            ),
            unsafe_allow_html=True
        )

        # --- 1. Agregar columna Categoría al DataFrame tabla para el gráfico ---
        tabla["Categoría"] = tabla["SKU"].apply(lambda x: PRODUCT_CATEGORY.get(int(x.replace("#", "")), "General"))

        # --- 2. Crear Layout de 2 Columnas para Gráficos en Tab 2 ---
        col_graf_izq, col_graf_der = st.columns(2)

        with col_graf_izq:
            # Gráfico de Torta (Doughnut) de distribución de pedidos por categoría
            df_cat = tabla.groupby("Categoría")["Sugerencia de Pedido"].sum().reset_index()
            df_cat_filt = df_cat[df_cat["Sugerencia de Pedido"] > 0]
            
            fig_pie = go.Figure()
            if not df_cat_filt.empty:
                fig_pie.add_trace(go.Pie(
                    labels=df_cat_filt["Categoría"],
                    values=df_cat_filt["Sugerencia de Pedido"],
                    hole=0.4,
                    marker=dict(colors=["#3B82F6", "#8B5CF6", "#10B981", "#F59E0B", "#EF4444", "#64748B"]),
                    textinfo="percent",
                    insidetextorientation="horizontal"
                ))
            else:
                fig_pie.add_trace(go.Pie(
                    labels=["Inventario Cubierto"],
                    values=[1],
                    hole=0.4,
                    marker=dict(colors=["#DEF7EC"]),
                    textinfo="label"
                ))
            fig_pie.update_layout(
                title="Distribución de Pedidos por Categoría",
                height=320,
                margin=dict(l=10, r=10, t=50, b=10),
                legend=dict(orientation="h", y=-0.1, font=dict(color="#475569")),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter, sans-serif", color="#1E293B")
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_graf_der:
            # Gráfico de Barras horizontales del Top 5 de productos con mayor demanda
            df_top5 = tabla.sort_values("Demanda Proyectada", ascending=False).head(5)
            
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                y=df_top5["Producto"],
                x=df_top5["Demanda Proyectada"],
                orientation="h",
                marker_color="#8B5CF6",
                text=df_top5["Demanda Proyectada"].round().astype(int),
                textposition="auto",
                name="Demanda Proyectada"
            ))
            fig_bar.update_layout(
                title="Top 5 Productos de Mayor Demanda Proyectada",
                height=320,
                margin=dict(l=10, r=10, t=50, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter, sans-serif", color="#1E293B"),
                xaxis=dict(gridcolor="#F1F5F9", tickfont=dict(color="#475569")),
                yaxis=dict(gridcolor="#F1F5F9", tickfont=dict(color="#475569"), autorange="reversed")
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # Tabla HTML/CSS interactiva con SKUs, categorías, y badges
        table_rows_html = ""
        for index, row in tabla.iterrows():
            item_id = int(row["SKU"].replace("#", ""))
            cat = PRODUCT_CATEGORY.get(item_id, "General")
            sku_code = PRODUCT_SKU.get(item_id, f"SKU-{item_id}")
            sug = row["Sugerencia de Pedido"]
            
            if sug > 0:
                badge_html = f'<span style="background: #FEF3C7; color: #92400E; font-size: 0.8rem; font-weight: 600; padding: 4px 12px; border-radius: 9999px; display: inline-block;">Pedir {sug} uds.</span>'
            else:
                badge_html = '<span style="background: #DEF7EC; color: #03543F; font-size: 0.8rem; font-weight: 600; padding: 4px 12px; border-radius: 9999px; display: inline-block;">Cubierto</span>'
                
            table_rows_html += (
                f'<tr style="border-bottom: 1px solid #F1F5F9; font-size: 0.9rem; color: #334155;">'
                f'<td style="padding: 14px 20px; font-weight: 500; color: #1E293B;">'
                f'{row["Producto"]}<br><span style="font-size: 0.75rem; color: #94A3B8;">{sku_code}</span>'
                f'</td>'
                f'<td style="padding: 14px 20px; color: #64748B;">{cat}</td>'
                f'<td style="padding: 14px 20px; text-align: center; font-weight: 600;">{row["Stock Actual"]}</td>'
                f'<td style="padding: 14px 20px; text-align: center; color: #8B5CF6; font-weight: 600;">'
                f'<span style="display: inline-flex; align-items: center; gap: 4px;">'
                f'<svg style="width: 14px; height: 14px; color: #8B5CF6;" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">'
                f'<path stroke-linecap="round" stroke-linejoin="round" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />'
                f'</svg>'
                f'{row["Demanda Proyectada"]}'
                f'</span>'
                f'</td>'
                f'<td style="padding: 14px 20px; text-align: center; color: #64748B;">{row["Stock Seguridad"]}</td>'
                f'<td style="padding: 14px 20px; text-align: center;">{badge_html}</td>'
                f'</tr>'
            )

        table_html = textwrap.dedent(
            f"""<div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.05); margin-top: 16px;">
                <div style="padding: 16px 20px; border-bottom: 1px solid #E2E8F0; font-weight: 600; color: #1E293B; font-size: 1rem;">Desglose General</div>
                <div style="overflow-x: auto;">
                    <table style="width: 100%; border-collapse: collapse; text-align: left; min-width: 800px;">
                        <thead>
                            <tr style="background: #F8FAFC; border-bottom: 1px solid #E2E8F0;">
                                <th style="padding: 14px 20px; font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase;">Producto / SKU</th>
                                <th style="padding: 14px 20px; font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase;">Categoría</th>
                                <th style="padding: 14px 20px; font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase; text-align: center;">Stock Actual</th>
                                <th style="padding: 14px 20px; font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase; text-align: center;">Demanda Proyectada</th>
                                <th style="padding: 14px 20px; font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase; text-align: center;">Stock Seguridad</th>
                                <th style="padding: 14px 20px; font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase; text-align: center;">Sugerencia de Pedido</th>
                            </tr>
                        </thead>
                        <tbody>{table_rows_html}</tbody>
                    </table>
                </div>
            </div>"""
        )
        st.markdown(table_html, unsafe_allow_html=True)

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        csv_general = tabla.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Descargar Reporte General Completo (CSV)",
            data=csv_general,
            file_name="reporte_general_compras.csv",
            mime="text/csv",
            key="btn_download_general",
            use_container_width=True
        )

st.markdown("---")
st.caption("Prototipo académico — Inteligencia Artificial aplicada a Series de Tiempo · Modelo: XGBoost\n\n"
           "Desarrollado por : Daniela Auquilla, José Salamea, Pedro Gonzalez y Carlos Moyano")