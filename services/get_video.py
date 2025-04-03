import io
import os
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account
from googleapiclient.discovery import build

creds = service_account.Credentials.from_service_account_file(
    './drive_secret.json',
    scopes=['https://www.googleapis.com/auth/drive']
)

drive_service = build('drive', 'v3', credentials=creds)

def get_video(file_id, directorio_salida='.'):
    metadata = drive_service.files().get(fileId=file_id).execute()
    nombre_archivo = metadata.get("name", "archivo_descargado")

    ruta_salida = os.path.join(directorio_salida, nombre_archivo)

    request = drive_service.files().get_media(fileId=file_id)
    with io.FileIO(ruta_salida, mode='wb') as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            if status:
                print(f"Descargando: {int(status.progress() * 100)}% completado")

    print(f"Archivo guardado en: {ruta_salida}")
    return ruta_salida
