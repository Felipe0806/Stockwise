import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from datetime import datetime, timedelta
from prophet import Prophet
from xgboost import XGBRegressor
import numpy as np

# 📌 Crear la API
app = FastAPI()

# 📌 Cargar todos los modelos al iniciar la API
modelos_prophet = {}
modelos_xgb = {}
modelos_path = "modelos_productos/"

if not os.path.exists(modelos_path):
    raise Exception("⚠️ No se encontró la carpeta de modelos.")

print("📂 Cargando modelos...")
for archivo in os.listdir(modelos_path):
    if archivo.endswith(".pkl"):
        try:
            if "modelo_xgb_" in archivo:
                producto = archivo.replace("modelo_xgb_", "").replace(".pkl", "")
                modelos_xgb[producto] = joblib.load(os.path.join(modelos_path, archivo))
                print(f"✅ Modelo XGBoost cargado: {archivo}")
            elif "modelo_prophet_" in archivo:
                producto = archivo.replace("modelo_prophet_", "").replace(".pkl", "")
                modelos_prophet[producto] = joblib.load(os.path.join(modelos_path, archivo))
                print(f"✅ Modelo Prophet cargado: {archivo}")
        except Exception as e:
            print(f"❌ Error al cargar {archivo}: {str(e)}")

if not modelos_xgb or not modelos_prophet:
    raise Exception("⚠️ No se pudo cargar ningún modelo Prophet o XGBoost.")

print("✅ Todos los modelos han sido cargados correctamente.")

# 📌 Endpoint para predecir ventas
@app.get("/predecir/{codigo_producto}")
async def predecir_ventas(codigo_producto: str, periodo: str = Query("semanal", enum=["semanal", "mensual"])):
    if codigo_producto not in modelos_xgb or codigo_producto not in modelos_prophet:
        raise HTTPException(status_code=404, detail="Modelo no encontrado para el producto.")

    modelo_prophet = modelos_prophet[codigo_producto]
    modelo_xgb = modelos_xgb[codigo_producto]

    # 📌 Crear fechas futuras (próximos 4 meses - 120 días)
    fecha_actual = datetime.today()
    dias_futuros = 120
    fechas_futuras = [fecha_actual + timedelta(days=i) for i in range(dias_futuros)]

    # 📌 Crear dataframe futuro con la columna 'evento_especial'
    futuro = pd.DataFrame({"ds": fechas_futuras})
    futuro["evento_especial"] = 0  # Se asume que no hay eventos especiales en fechas futuras

    # 📌 Predecir con Prophet
    predicciones_prophet = modelo_prophet.predict(futuro)
    futuro["yhat"] = predicciones_prophet["yhat"]

    # 📌 Generar características para XGBoost
    futuro["lag_1"] = futuro["yhat"].shift(1).fillna(method='bfill')
    futuro["lag_7"] = futuro["yhat"].shift(7).fillna(method='bfill')
    futuro["lag_30"] = futuro["yhat"].shift(30).fillna(method='bfill')
    futuro["rolling_mean_7"] = futuro["yhat"].rolling(7, min_periods=1).mean()
    futuro["rolling_mean_30"] = futuro["yhat"].rolling(30, min_periods=1).mean()
    futuro.fillna(0, inplace=True)

    # 📌 Predecir residuales con XGBoost
    features = ["evento_especial", "lag_1", "lag_7", "lag_30", "rolling_mean_7", "rolling_mean_30"]
    predicciones_xgb = modelo_xgb.predict(futuro[features])
    futuro["cantidad_predicha"] = futuro["yhat"] + predicciones_xgb

    # 📌 Asegurar que las predicciones no sean negativas
    futuro["cantidad_predicha"] = futuro["cantidad_predicha"].clip(lower=0)

    # 📌 Agrupar predicciones según el periodo elegido
    if periodo == "semanal":
        futuro["periodo"] = futuro["ds"].dt.strftime("%Y-%U")  # Año-Semana
    elif periodo == "mensual":
        futuro["periodo"] = futuro["ds"].dt.strftime("%Y-%m")  # Año-Mes

    resultado = futuro.groupby("periodo")["cantidad_predicha"].sum().reset_index()

    return resultado.to_dict(orient="records")

# 📌 Iniciar la API
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
