import os
import json
import pandas as pd

# Carpeta de precios
prices_dir = r"C:\Users\57305\Documents\FTZ_model_2.0\simulations\ga_benchmark\data\prices"

# Lista donde iremos acumulando cada entorno
rows = []

for filename in os.listdir(prices_dir):
    if filename.endswith("_prices.json"):
        path = os.path.join(prices_dir, filename)
        with open(path, "r") as f:
            data = json.load(f)

        # Quitar sufijo "_prices.json" → env_id
        env_id = filename.replace("_prices.json", "")

        # Cada archivo puede tener distintas estructuras.
        # Si es un diccionario anidado, lo aplanamos.
        def flatten_json(y, prefix=""):
            out = {}
            if isinstance(y, dict):
                for k, v in y.items():
                    out.update(flatten_json(v, f"{prefix}{k}_"))
            elif isinstance(y, list):
                for i, v in enumerate(y):
                    out.update(flatten_json(v, f"{prefix}{i}_"))
            else:
                out[prefix[:-1]] = y
            return out

        flat_data = flatten_json(data)
        flat_data["env_id"] = env_id
        rows.append(flat_data)

# Crear el DataFrame consolidado
df_prices = pd.DataFrame(rows)

# Reordenar columnas (env_id primero)
cols = ["env_id"] + [c for c in df_prices.columns if c != "env_id"]
df_prices = df_prices[cols]

# Mostrar resumen
print(f"✅ DataFrame creado con {df_prices.shape[0]} entornos y {df_prices.shape[1]} columnas")
df_prices.head()
