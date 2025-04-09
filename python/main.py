import json
import os
from services.drive import (
    authenticate_drive,
    get_video,
    get_parent_folder,
    upload_video,
    # copiar_permisos
)
from services.stabilizer import video_stabilizer

def procesar_videos(video_index_path, directorio_videos, output_json):
    # Autenticar en Drive con PyDrive
    drive = authenticate_drive()

    # Cargar el índice de videos (videoStableIndex.json)
    with open(video_index_path, "r") as f:
        videos_index = json.load(f)
    
    # Usaremos el mismo diccionario para actualizar en línea
    resultados = videos_index
    
    # Recorrer cada grupo (r1, r2, etc.)
    for grupo, horarios in videos_index.items():
        # Cada grupo contiene varios horarios (h1, h2, h3, h4)
        for horario, dias in horarios.items():
            # Cada horario contiene los días: m, t y w
            for dia, video_info in dias.items():
                original_id = video_info.get("original", "")
                stable_value = video_info.get("stable", "")
                
                # Omitir si no hay ID original o ya se procesó (stable no vacío)
                if not original_id:
                    continue
                if stable_value:
                    print(f"Saltando video {grupo} {horario} {dia} (ya tiene estable).")
                    continue

                print(f"Procesando video {grupo} {horario} {dia} ...")
                
                # 1) Descargar el video original usando el ID original
                ruta_local = get_video(drive, original_id, directorio_videos)
                
                # 2) Estabilizar el video y crear la ruta para el video estabilizado
                ruta_stable = ruta_local.upper().replace('.MP4', '_STABLE.MP4')
                video_stabilizer(ruta_local, ruta_stable)
                
                # 3) Obtener la carpeta padre del archivo original en Drive
                carpeta_padre = get_parent_folder(drive, original_id)
                
                # 4) Subir el video estabilizado a la misma carpeta en Drive
                new_stable_id = upload_video(drive, ruta_stable, carpeta_padre)
                
                # 5) Guardar el ID estabilizado en el campo correspondiente
                resultados[grupo][horario][dia]["stable"] = new_stable_id
                
                # Eliminar los archivos locales para liberar espacio
                os.remove(ruta_local)
                os.remove(ruta_stable)
                print(f"Video {grupo} {horario} {dia} procesado y subido con ID: {new_stable_id}")
                
                # Actualizar el archivo JSON de forma incremental
                with open(output_json, "w") as f:
                    json.dump(resultados, f, indent=4)
                print("Archivo JSON actualizado en:", output_json)
    
    print("Procesamiento completado.")

if __name__ == '__main__':
    # Ahora se utiliza videoStableIndex.json tanto para entrada como para salida
    video_index = "./videoStableIndex.json" 
    directorio_videos = "./videos"
    output_json = "videoStableIndex.json"
    procesar_videos(video_index, directorio_videos, output_json)
