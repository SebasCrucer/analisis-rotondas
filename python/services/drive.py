import os
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

def authenticate_drive():
    """
    Autentica a Google Drive utilizando PyDrive.
    Se abrirá un navegador para completar la autenticación.
    Devuelve el objeto GoogleDrive autenticado.
    """
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()
    drive = GoogleDrive(gauth)
    return drive

def get_video(drive, file_id, directorio_salida='.'):
    file_obj = drive.CreateFile({'id': file_id})
    # Solicita 'title', 'downloadUrl' y 'parents' en los metadatos
    file_obj.FetchMetadata(fields='title,downloadUrl,parents')
    
    # Verifica que el archivo sea descargable
    if not file_obj.get('downloadUrl'):
        raise Exception("El archivo no es descargable. Verifica que no sea un archivo nativo de Google (Docs, Sheets, etc.).")
    
    file_name = file_obj.get('title', 'archivo_descargado')
    ruta_salida = os.path.join(directorio_salida, file_name)
    file_obj.GetContentFile(ruta_salida)
    print(f"Archivo guardado en: {ruta_salida}")
    return ruta_salida

def get_parent_folder(drive, file_id):
    """
    Obtiene el ID de la carpeta padre del archivo identificado por file_id.
    Si no tiene carpeta padre, devuelve None.
    """
    file_obj = drive.CreateFile({'id': file_id})
    file_obj.FetchMetadata(fields='parents')
    padres = file_obj.get('parents', [])
    if padres:
        # Cada elemento es un diccionario con la clave "id"
        return padres[0]['id'] if isinstance(padres[0], dict) else padres[0]
    else:
        return None

def upload_video(drive, file_path, parent_folder_id):
    """
    Sube un archivo a Google Drive. Si se especifica parent_folder_id, el archivo se
    ubica dentro de esa carpeta.
    Devuelve el ID del archivo subido.
    """
    file_name = os.path.basename(file_path)
    metadata = {'title': file_name}
    if parent_folder_id:
        metadata['parents'] = [{'id': parent_folder_id}]
    file_obj = drive.CreateFile(metadata)
    file_obj.SetContentFile(file_path)
    file_obj.Upload()
    print(f"Video {file_name} subido con ID: {file_obj['id']}")
    return file_obj['id']

def copiar_permisos(drive, original_id, nuevo_id):
    """
    Copia los permisos del archivo original (original_id) al nuevo archivo (nuevo_id).
    Se omite el permiso del propietario para evitar duplicaciones.
    Para permisos de tipo 'user' o 'group' se requiere incluir el campo emailAddress.
    """
    # Utilizamos el servicio subyacente de PyDrive para acceder a la API de Drive
    service = drive.auth.service
    permisos = service.permissions().list(
        fileId=original_id,
        fields="permissions"
    ).execute()
    for permiso in permisos.get("permissions", []):
        if permiso.get("role") == "owner":
            continue
        body_permiso = {
            "type": permiso.get("type"),
            "role": permiso.get("role")
        }
        if permiso.get("type") in ["user", "group"]:
            body_permiso["emailAddress"] = permiso.get("emailAddress")
        service.permissions().create(
            fileId=nuevo_id,
            body=body_permiso
        ).execute()
