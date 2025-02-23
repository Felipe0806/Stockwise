import pandas as pd
import os
import joblib
from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# Para evitar problemas con backends de matplotlib
import matplotlib
matplotlib.use("Agg")

# 1. Cargar datos
archivo_csv = "dataset_completo.csv"
df = pd.read_csv(archivo_csv, parse_dates=["fecha"], dtype={"producto": str})

# 2. Ajustar nombres de columnas
df.rename(columns={"fecha": "ds", "cantidad": "y"}, inplace=True)

# 3. Convertir fechas y filtrar rango
df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
df.dropna(subset=["ds"], inplace=True)
df = df[(df["ds"].dt.year >= 2021) & (df["ds"].dt.year <= 2025)]

# 4. Limpiar datos (valores nulos o cero)
df.dropna(subset=["y"], inplace=True)
df = df[df["y"] > 0]

# Obtener lista de productos únicos
productos_unicos = df["producto"].unique()

for producto_especifico in productos_unicos:
    df_producto = df[df["producto"] == producto_especifico].copy()

    if df_producto.empty:
        print(f"⚠ El producto {producto_especifico} no se encuentra en los datos.")
        continue
    
    df_producto = df_producto.groupby("ds").agg({
        "y": "sum",
        "evento_especial": "max"
    }).reset_index()

    df_producto.set_index("ds", inplace=True)
    df_producto.sort_index(inplace=True)

    min_fecha, max_fecha = df_producto.index.min(), df_producto.index.max()
    idx = pd.date_range(min_fecha, max_fecha, freq='D')

    df_producto = df_producto.reindex(idx, fill_value=0)
    df_producto.index.name = 'ds'
    df_producto.reset_index(inplace=True)

    if len(df_producto) < 10:
        print(f"⚠ Datos insuficientes después de la limpieza para {producto_especifico}")
        continue

    for lag in [1, 7, 30]:
        df_producto[f"lag_{lag}"] = df_producto["y"].shift(lag)
    df_producto["rolling_mean_7"] = df_producto["y"].rolling(7).mean()
    df_producto["rolling_mean_30"] = df_producto["y"].rolling(30).mean()
    df_producto.dropna(inplace=True)

    train_size = int(len(df_producto) * 0.70)
    df_train = df_producto.iloc[:train_size].copy()
    df_test = df_producto.iloc[train_size:].copy()

    modelo_prophet = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='multiplicative',
        changepoint_prior_scale=0.001,
        seasonality_prior_scale=50
    )
    modelo_prophet.add_regressor("evento_especial")
    modelo_prophet.fit(df_train)

    futuro = modelo_prophet.make_future_dataframe(periods=len(df_test), freq='D')
    futuro = futuro.merge(df_test[["ds", "evento_especial"]], on="ds", how="left").fillna(0)
    predicciones_prophet = modelo_prophet.predict(futuro)
    
    df_test = df_test.merge(predicciones_prophet[["ds", "yhat"]], on="ds", how="left")
    df_train["residual"] = df_train["y"] - modelo_prophet.predict(df_train)["yhat"].values
    df_test["residual"] = df_test["y"] - df_test["yhat"]

    features = [
        "evento_especial",
        "lag_1", "lag_7", "lag_30",
        "rolling_mean_7", "rolling_mean_30"
    ]

    modelo_xgb = XGBRegressor(
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=50
    )
    modelo_xgb.fit(df_train[features], df_train["residual"])

    predicciones_xgb = modelo_xgb.predict(df_test[features])
    df_test["prediccion_final"] = df_test["yhat"] + predicciones_xgb

    mae = mean_absolute_error(df_test["y"], df_test["prediccion_final"])
    rmse = np.sqrt(mean_squared_error(df_test["y"], df_test["prediccion_final"]))
    r2 = r2_score(df_test["y"], df_test["prediccion_final"])

    print(f"Producto {producto_especifico} - ⚡ MAE: {mae}, 📉 RMSE: {rmse}, 📊 R²: {r2}")

    if r2 > 0.70:

        os.makedirs("modelos_productos", exist_ok=True)
        ruta_modelo_prophet = f"modelos_productos/modelo_prophet_{producto_especifico}.pkl"
        joblib.dump(modelo_prophet, ruta_modelo_prophet)
        print(f"✅ Modelo Prophet guardado en {ruta_modelo_prophet}")

        os.makedirs("modelos_productos", exist_ok=True)
        ruta_modelo_xgb = f"modelos_productos/modelo_xgb_{producto_especifico}.pkl"
        joblib.dump(modelo_xgb, ruta_modelo_xgb)
        print(f"✅ Modelo XGBoost guardado en {ruta_modelo_xgb}")

        plt.figure(figsize=(15, 7))
        plt.plot(df_train["ds"], df_train["y"], label="Entrenamiento", color="blue", alpha=0.7)
        plt.plot(df_test["ds"], df_test["y"], label="Real", color="green", alpha=0.7)
        plt.plot(df_test["ds"], df_test["prediccion_final"], label="Predicción Prophet + XGBoost", color="red", linestyle="dashed", alpha=0.8)
        
        plt.xlabel("Fecha")
        plt.ylabel("Cantidad")
        plt.title(f"Predicción de ventas para producto {producto_especifico}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"modelos_productos/prediccion_{producto_especifico}.png", dpi=300, bbox_inches='tight')
        plt.close()

print("🎉 Entrenamiento completado!")
