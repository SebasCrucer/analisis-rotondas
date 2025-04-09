import cv2
import numpy as np

def video_stabilizer(ruta_entrada, 
                     ruta_salida=None, 
                     max_features=200, 
                     quality_level=0.01, 
                     min_distance=15):
    """
    Estabiliza un video anclándolo al primer frame con deformación (homografía fija).
    """
    if ruta_salida is None:
        ruta_salida = ruta_entrada.replace('.mp4', '_stable.mp4')

    cap = cv2.VideoCapture(ruta_entrada)
    if not cap.isOpened():
        print(f"No se pudo abrir el video: {ruta_entrada}")
        return

    # Leer el primer frame
    ret, primer_frame = cap.read()
    if not ret:
        print("No se pudo leer el primer frame. Abortando.")
        return

    # Obtener dimensiones reales del primer frame
    alto, ancho = primer_frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Crear VideoWriter con la resolución obtenida del primer frame
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(ruta_salida, fourcc, fps, (ancho, alto))

    # Convertir el primer frame a escala de grises
    primer_gray = cv2.cvtColor(primer_frame, cv2.COLOR_BGR2GRAY)

    # 1) Detectar características en el primer frame
    puntos_iniciales = cv2.goodFeaturesToTrack(
        primer_gray,
        maxCorners=max_features,
        qualityLevel=quality_level,
        minDistance=min_distance
    )

    if puntos_iniciales is None or len(puntos_iniciales) < 4:
        print("No se detectaron suficientes puntos en el primer frame. Abortando.")
        return

    # Guardamos una copia de estos puntos como referencia
    puntos_ref = np.copy(puntos_iniciales)

    # Parámetros para calcOpticalFlowPyrLK
    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    )

    frame_anterior_gray = primer_gray.copy()

    while True:
        ret, frame_actual = cap.read()
        if not ret:
            break

        frame_actual_gray = cv2.cvtColor(frame_actual, cv2.COLOR_BGR2GRAY)

        # 2) Rastrear los puntos en el frame actual
        puntos_nuevos, status, _ = cv2.calcOpticalFlowPyrLK(
            frame_anterior_gray,
            frame_actual_gray,
            puntos_iniciales,
            None,
            **lk_params
        )

        if puntos_nuevos is None or status is None:
            out.write(frame_actual)
            frame_anterior_gray = frame_actual_gray
            continue

        status = status.flatten()

        if len(status) != len(puntos_iniciales) or len(status) != len(puntos_nuevos):
            out.write(frame_actual)
            frame_anterior_gray = frame_actual_gray
            continue

        # Seleccionar los puntos rastreados con éxito
        puntos_buenos = (status == 1)
        puntos_iniciales_filtrados = puntos_iniciales[puntos_buenos]
        puntos_nuevos_filtrados = puntos_nuevos[puntos_buenos]
        puntos_ref_filtrados = puntos_ref[puntos_buenos]

        if len(puntos_iniciales_filtrados) < 4:
            out.write(frame_actual)
        else:
            # 3) Calcular la homografía
            H, _ = cv2.findHomography(
                puntos_nuevos_filtrados.reshape(-1, 1, 2),
                puntos_ref_filtrados.reshape(-1, 1, 2),
                cv2.RANSAC,
                5.0
            )

            if H is not None:
                # 4) Aplicar la transformación al frame actual
                frame_estabilizado = cv2.warpPerspective(frame_actual, H, (ancho, alto))
                out.write(frame_estabilizado)
            else:
                out.write(frame_actual)

        frame_anterior_gray = frame_actual_gray.copy()
        puntos_iniciales = puntos_nuevos_filtrados.reshape(-1, 1, 2)
        puntos_ref = puntos_ref_filtrados.reshape(-1, 1, 2)

    cap.release()
    out.release()
    print(f"Video estabilizado guardado en: {ruta_salida}")
