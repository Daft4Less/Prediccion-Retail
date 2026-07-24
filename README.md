# Sistema de Predicción de Demanda y Gestión de Inventario — IA para Retail

Este repositorio contiene una aplicación web interactiva desarrollada en **Streamlit** que utiliza modelos predictivos de **Machine Learning (XGBoost)** para anticipar la demanda de productos y optimizar las sugerencias de órdenes de compra en el sector retail.

---

##  Video de Demostración
Haz clic en la imagen a continuación para ver el video explicativo de la aplicación en YouTube:

[![Presentación del Panel Predictivo](https://img.youtube.com/vi/qil-y4AQN30/maxresdefault.jpg)](https://youtu.be/qil-y4AQN30)

---

##  Estructura del Repositorio
* **`app.py`**: Código fuente de la interfaz gráfica y la lógica del negocio.
* **`requirements.txt`**: Librerías y dependencias necesarias para ejecutar el proyecto.
* **`ejecutar.txt`**: Guía rápida sintetizada para despliegues independientes en otras computadoras.
* **`modelo/`**: Carpeta que almacena el archivo `modelo.onnx` (modelo de IA pre-entrenado) y su metadata.
* **`data/`**: Contiene archivos de demostración histórica de ventas e inventarios.

---

##  Arquitectura e Ingeniería de la IA

### 1. Preparación de Características (Feature Engineering)
El modelo predice la demanda del día de mañana basándose en la serie temporal histórica del producto. Para ello, se calculan características dinámicas:
* **Variables Rezagadas (Lags)**: Ventas de 1, 2 y 7 días previos.
* **Ventanas Móviles (Rolling Features)**: Promedios de ventas de los últimos 7 y 30 días para suavizar la curva y detectar tendencias recientes de demanda.

### 2. Algoritmo y Entrenamiento
* **Algoritmo**: XGBoost Regressor (Árboles de Decisión Potenciados), que destaca por capturar patrones complejos no lineales en series de tiempo de retail.
* **Error Métrico**: Error Absoluto Medio (MAE) de aproximadamente **8 unidades**, lo que representa un margen de error mínimo sobre el promedio de ventas diarias del catálogo.

### 3. Comunicación con el Modelo (Serialización ONNX)
Para que el sistema sea extremadamente rápido y no requiera compilar compiladores de C++ de XGBoost en producción, el modelo se entrenó en un Notebook de Jupyter (`Prototipo_V2_Retail.ipynb`) y se exportó a formato **ONNX**. 
En `app.py`, nos comunicamos con este archivo usando **ONNX Runtime** a través de la clase envoltorio `ModeloONNX`:
* Carga el archivo `modelo/modelo.onnx` en memoria en menos de 50 ms.
* Recibe un vector de entrada NumPy con las características del producto y ejecuta inferencias en CPU en menos de 10 ms mediante `.predict(X)`.

---

##  Explicación de Funciones Principales en `app.py`

### 1. `ModeloONNX` (Línea 21)
* **Propósito**: Envuelve el archivo `.onnx` para que funcione como un modelo clásico de Scikit-Learn.
* **Método `predict(X)`**: Convierte el set de datos en arreglos float32 y ejecuta `sesion.run` para obtener el vector de predicción de ventas.

### 2. `cargar_modelo_preentrenado()` (Línea 335)
* **Propósito**: Carga la red neuronal ONNX predefinida en memoria junto con su archivo de configuración JSON que contiene las variables y listado de productos activos.

### 3. `predecir_demanda_rango()` (Línea 403)
* **Propósito**: Ejecuta predicciones multi-día a futuro de forma recursiva.
* **Funcionamiento**: Calcula los lags y rolling means del día objetivo, predice la venta, y retroalimenta recursivamente el set de datos para predecir el día siguiente dentro del rango temporal seleccionado.

### 4. `calcular_compra_sugerida()` (Línea 478)
* **Propósito**: Aplica la fórmula de negocio para compras de reabastecimiento:
  $$\text{Cantidad a Pedir} = \max(0, \text{Venta Proyectada} + \text{Stock de Seguridad} - \text{Stock Actual})$$
* Evita valores negativos de compras si el stock actual cubre con creces las ventas esperadas.

---

##  Guía de Instalación y Ejecución

1. **Clonar o descargar** la carpeta del proyecto en tu computadora.
2. **Abrir la consola** (CMD o Terminal) y navegar a la carpeta raíz del proyecto.
3. **Instalar dependencias** con `pip`:
   ```bash
   pip install -r requirements.txt
   ```
4. **Verificar el modelo de IA** (Paso opcional para demostración en vivo antes de abrir el dashboard):
   * **Opción A (Ejecutar script interactivo)**:
     ```bash
     py verificar_modelo.py
     ```
   * **Opción B (Comando directo de una sola línea)**:
     ```bash
     py -c "import pandas as pd, json; m=json.load(open('modelo/metadata.json')); h=pd.read_parquet('modelo/historico_features.parquet'); print('Modelo cargado correctamente'); print('Tipo de modelo: ModeloONNX'); print('Error de validacion (MAE):', round(m['mae'], 2), 'unidades'); print('Registros del historico usado:', len(h)); print('Variables que usa el modelo:', len(m['columnas_features']))"
     ```
     *(Debería imprimir en consola el estado correcto de validación de 5.33 MAE y las 15 variables de entrada).*

5. **Correr la aplicación** con Streamlit:
   ```bash
   py -m streamlit run app.py
   ```
6. Acceder automáticamente en tu navegador a **[http://localhost:8501](http://localhost:8501)**.

---

## 👥 Desarrolladores
* **Daniela Auquilla** (Project Manager)
* **José Salamea** 
* **Pedro Gonzalez** 
* **Carlos Moyano** 
