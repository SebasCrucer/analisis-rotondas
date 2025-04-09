import os
import pandas as pd

# Ruta a la carpeta que contiene los CSV
folder = './merge/csvs'

# Lista todos los archivos CSV en la carpeta
file_paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.csv')]

# Mapeo para archivos tipo "t"
mapping_t = {
    "id": "id",
    "r_id": "rotonda",
    "h_id": "horario",
    # Se elimina la columna 'd'
    "Id_in": "entrada",
    "Id_out": "salida",
    "in_timestamp": "tiempo_entrada",
    "out_timestamp": "tiempo_salida",
    "duration": "duracion",
    "type": "tipo"
}

# Mapeo para archivos tipo "m" o "w"
mapping_m = {
    "Id": "id",
    "Num rotonda": "rotonda",
    "horario": "horario",
    "dia": "dia",
    "Id_Entrada": "entrada",
    "Id_salida": "salida",
    "Tiempo Entrada": "tiempo_entrada",
    "Tiempo Salida": "tiempo_salida",
    "Tiempo dentro": "duracion",
    "Tipo vehículo": "tipo"
}

# Lista para almacenar los DataFrames procesados
dfs = []

for path in file_paths:
    df = pd.read_csv(path)
    
    # Determinar la estructura del archivo y renombrar las columnas
    if "id" in df.columns and "r_id" in df.columns:  # Archivo tipo "t"
        df = df.rename(columns=mapping_t)
        if "d" in df.columns:
            df = df.drop(columns=["d"])
    elif "Id" in df.columns and "Num rotonda" in df.columns:  # Archivo tipo "m" o "w"
        df = df.rename(columns=mapping_m)
    else:
        print(f"Estructura no reconocida en {path}")
        continue

    # Extraer la letra que representa el día desde el nombre del archivo
    # Ejemplo: "r1-h4-t.csv" => "t"
    filename = os.path.basename(path)
    day_letter = filename.split("-")[-1].split(".")[0]
    # Asignar (o sobrescribir) la columna "dia" con la letra correspondiente
    df["dia"] = day_letter

    # Extraer el horario desde el nombre del archivo
    # Ejemplo: "r1-h4-t.csv" => "h4"
    horario = filename.split("-")[1]
    df["horario"] = horario

    # Extraer el ID de la rotonda desde el nombre del archivo
    # Ejemplo: "r1-h4-t.csv" => "r1"
    rotonda = filename.split("-")[0]
    df["rotonda"] = rotonda

    dfs.append(df)

# Concatenar todos los DataFrames en uno solo
df_merged = pd.concat(dfs, ignore_index=True)

# Guardar el DataFrame unido en un archivo CSV
df_merged.to_csv("merged.csv", index=False)
print("CSV 'merged.csv' guardado correctamente.")
