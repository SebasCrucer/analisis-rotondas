"""
Módulo para el procesamiento automático de videos.

Este módulo implementa la funcionalidad para procesar automáticamente todos los videos
estables del índice, utilizando el contador de vehículos y la selección visual de zonas.
"""

import os
import json
import csv
import cv2
import numpy as np
from tqdm import tqdm
import gc
from concurrent.futures import ThreadPoolExecutor
import supervision as sv

from services.drive import authenticate_drive, get_video
from services.javat import VideoProcessor, ZONE_IN_POLYGONS, ZONE_OUT_POLYGONS
from services.zone_selector import ZoneSelector

class AutomaticVideoProcessor:
    """
    Clase para el procesamiento automático de videos estables.
    """
    def __init__(self, video_index_path="./videoStableIndex.json", output_dir="./resultados"):
        """
        Inicializa el procesador automático de videos.
        
        Args:
            video_index_path: Ruta al archivo JSON con el índice de videos estables
            output_dir: Directorio donde se guardarán los resultados
        """
        self.video_index_path = video_index_path
        self.output_dir = output_dir
        self.csv_path = os.path.join(output_dir, "vehiculos_contados.csv")
        self.temp_dir = os.path.join(output_dir, "temp_videos")
        
        # Crear directorios si no existen
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Inicializar CSV con encabezados
        self.initialize_csv()
    
    def initialize_csv(self):
        """Inicializa el archivo CSV con los encabezados requeridos."""
        with open(self.csv_path, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([
                "Id", 
                "rotonda (r)", 
                "horario (h)", 
                "dia (m, t o w)", 
                "Id_Entrada", 
                "Id_salida", 
                "Tiempo Entrada", 
                "Tiempo Salida", 
                "Tipo vehículo", 
                "Color"
            ])
    
    def load_video_index(self):
        """
        Carga el índice de videos estables.
        
        Returns:
            dict: Índice de videos estables o None si hay error
        """
        try:
            with open(self.video_index_path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error al cargar el índice de videos: {e}")
            return None
    
    def process_all_videos(self, model_weights_path, confidence=0.3, iou=0.7):
        """
        Procesa todos los videos estables del índice.
        
        Args:
            model_weights_path: Ruta al archivo de pesos del modelo YOLO
            confidence: Umbral de confianza para las detecciones
            iou: Umbral de IOU para las detecciones
            
        Returns:
            bool: True si se procesaron todos los videos correctamente
        """
        # Cargar el índice de videos
        videos_index = self.load_video_index()
        if not videos_index:
            print("No se pudo cargar el índice de videos.")
            return False
        
        # Autenticar en Drive
        try:
            drive = authenticate_drive()
        except Exception as e:
            print(f"Error al autenticar en Drive: {e}")
            return False
        
        # Contador de videos procesados
        videos_procesados = 0
        videos_totales = self.count_stable_videos(videos_index)
        
        print(f"Se procesarán {videos_totales} videos estables.")
        
        # Recorrer cada grupo (r1, r2, etc.)
        for grupo, horarios in videos_index.items():
            # Cada grupo contiene varios horarios (h1, h2, h3, h4)
            for horario, dias in horarios.items():
                # Cada horario contiene los días: m, t y w
                for dia, video_info in dias.items():
                    stable_id = video_info.get("stable", "")
                    
                    # Omitir si no hay ID estable
                    if not stable_id:
                        print(f"Saltando video {grupo} {horario} {dia} (no tiene estable).")
                        continue
                    
                    videos_procesados += 1
                    print(f"Procesando video {videos_procesados}/{videos_totales}: {grupo} {horario} {dia} ...")
                    
                    # Procesar el video
                    success = self.process_single_video(
                        drive=drive,
                        stable_id=stable_id,
                        grupo=grupo,
                        horario=horario,
                        dia=dia,
                        model_weights_path=model_weights_path,
                        confidence=confidence,
                        iou=iou
                    )
                    
                    if not success:
                        print(f"Error al procesar el video {grupo} {horario} {dia}.")
        
        print(f"Procesamiento completado. Se procesaron {videos_procesados} videos.")
        print(f"Resultados guardados en: {self.csv_path}")
        return True
    
    def count_stable_videos(self, videos_index):
        """
        Cuenta el número de videos estables en el índice.
        
        Args:
            videos_index: Índice de videos
            
        Returns:
            int: Número de videos estables
        """
        count = 0
        for grupo, horarios in videos_index.items():
            for horario, dias in horarios.items():
                for dia, video_info in dias.items():
                    if video_info.get("stable", ""):
                        count += 1
        return count
    
    def process_single_video(self, drive, stable_id, grupo, horario, dia, model_weights_path, confidence=0.3, iou=0.7):
        """
        Procesa un solo video estable.
        
        Args:
            drive: Objeto GoogleDrive autenticado
            stable_id: ID del video estable en Drive
            grupo: Identificador de la rotonda (r1, r2, etc.)
            horario: Identificador del horario (h1, h2, etc.)
            dia: Identificador del día (m, t, w)
            model_weights_path: Ruta al archivo de pesos del modelo YOLO
            confidence: Umbral de confianza para las detecciones
            iou: Umbral de IOU para las detecciones
            
        Returns:
            bool: True si se procesó correctamente
        """
        try:
            # 1) Descargar el video estable
            video_path = get_video(drive, stable_id, self.temp_dir)
            if not video_path or not os.path.exists(video_path):
                print(f"Error: No se pudo descargar el video {stable_id}")
                return False
            
            # 2) Permitir al usuario seleccionar zonas de entrada y salida
            zone_selector = ZoneSelector(video_path)
            zones_in, zones_out, loaded = zone_selector.load_zones()
            
            if not loaded:
                print(f"Seleccionando zonas para el video {os.path.basename(video_path)}...")
                if not zone_selector.select_zones():
                    print(f"Error: No se pudieron seleccionar zonas para el video {os.path.basename(video_path)}")
                    if os.path.exists(video_path):
                        os.remove(video_path)
                    return False
                
                # Cargar las zonas recién seleccionadas
                zones_in, zones_out, loaded = zone_selector.load_zones()
                if not loaded:
                    print(f"Error: No se pudieron cargar las zonas seleccionadas")
                    if os.path.exists(video_path):
                        os.remove(video_path)
                    return False
            
            # Actualizar las zonas globales
            global ZONE_IN_POLYGONS, ZONE_OUT_POLYGONS
            ZONE_IN_POLYGONS = zones_in
            ZONE_OUT_POLYGONS = zones_out
            
            # 3) Procesar el video con el contador
            output_video = os.path.join(self.output_dir, f"{grupo}_{horario}_{dia}_processed.mp4")
            
            # Crear el procesador de video
            processor = VideoProcessor(
                source_weights_path=model_weights_path,
                source_video_path=video_path,
                target_video_path=output_video,
                confidence_threshold=confidence,
                iou_threshold=iou,
                num_rotonda=grupo,
                horario=horario,
                dia=dia,
            )
            
            # Procesar el video
            processor.process_video()
            
            # 4) Actualizar el CSV con los eventos del procesador
            self.update_csv_with_events(processor.detections_manager.events)
            
            # 5) Limpiar archivos temporales
            if os.path.exists(video_path):
                os.remove(video_path)
            
            print(f"Video {grupo} {horario} {dia} procesado correctamente.")
            return True
            
        except Exception as e:
            print(f"Error al procesar el video {grupo} {horario} {dia}: {e}")
            return False
    
    def update_csv_with_events(self, events):
        """
        Actualiza el archivo CSV con los eventos de vehículos detectados.
        
        Args:
            events: Lista de eventos de vehículos detectados
        """
        with open(self.csv_path, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            for event in events:
                writer.writerow([
                    event["Id"],
                    event["Num rotonda"],
                    event["horario"],
                    event["dia"],
                    event["Id_Entrada"],
                    event["Id_salida"],
                    event["Tiempo Entrada"],
                    event["Tiempo Salida"],
                    event["Tipo vehículo"],
                    ""  # Color (no disponible en la implementación actual)
                ])


class EnhancedVideoProcessor(VideoProcessor):
    """
    Versión mejorada del procesador de video que incluye detección de color.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tracker_id_to_color = {}
    
    def process_frame(self, frame, frame_time):
        """
        Procesa un frame utilizando YOLO, actualiza el tracker y asigna el tipo de vehículo y color.
        
        Args:
            frame: Frame a procesar
            frame_time: Tiempo del frame
            
        Returns:
            np.ndarray: Frame anotado
        """
        results = self.model(
            frame, verbose=False, conf=self.conf_threshold, iou=self.iou_threshold
        )[0]
        detections = sv.Detections.from_ultralytics(results)
        
        # Guarda los tipos de vehículo originales y sus coordenadas antes del tracking
        orig_vehicle_types = {}
        for i, cls in enumerate(detections.class_id):
            vehicle_type = self.model.names.get(int(cls), "unknown")
            if i < len(detections.xyxy):
                # Guardamos la posición para vincularla después del tracking
                box_key = tuple(map(float, detections.xyxy[i]))
                orig_vehicle_types[box_key] = vehicle_type
                
                # Detectar color del vehículo
                if i < len(detections.xyxy):
                    x1, y1, x2, y2 = map(int, detections.xyxy[i])
                    # Asegurarse de que las coordenadas estén dentro de los límites del frame
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(frame.shape[1], x2)
                    y2 = min(frame.shape[0], y2)
                    
                    if x2 > x1 and y2 > y1:
                        vehicle_roi = frame[y1:y2, x1:x2]
                        color_name = self.detect_vehicle_color(vehicle_roi)
                        # Guardar el color para asociarlo después del tracking
                        orig_vehicle_types[box_key] = (vehicle_type, color_name)
        
        # Actualiza el tracker
        detections = self.tracker.update_with_detections(detections)
        
        # Actualiza el mapeo de tracker_id a tipo de vehículo y color
        for i, tracker_id in enumerate(detections.tracker_id):
            if i < len(detections.xyxy):
                box_key = tuple(map(float, detections.xyxy[i]))
                # Intenta encontrar el tipo de vehículo y color original más cercano
                if box_key in orig_vehicle_types:
                    if isinstance(orig_vehicle_types[box_key], tuple):
                        vehicle_type, color_name = orig_vehicle_types[box_key]
                        self.tracker_id_to_vehicle_type[tracker_id] = vehicle_type
                        self.tracker_id_to_color[tracker_id] = color_name
                    else:
                        self.tracker_id_to_vehicle_type[tracker_id] = orig_vehicle_types[box_key]
                        self.tracker_id_to_color[tracker_id] = "Desconocido"
        
        # Asigna los tipos de vehículo usando nuestro diccionario
        vehicle_types = []
        vehicle_colors = []
        for tracker_id in detections.tracker_id:
            vehicle_types.append(self.tracker_id_to_vehicle_type.get(tracker_id, "Desconocido"))
            vehicle_colors.append(self.tracker_id_to_color.get(tracker_id, "Desconocido"))
        
        detections.vehicle_type = np.array(vehicle_types)
        detections.vehicle_color = np.array(vehicle_colors)
        
        # Procesa las detecciones en las zonas de entrada y salida
        detections_in_zones = []
        detections_out_zones = []
        for zone_in, zone_out in zip(self.zones_in, self.zones_out):
            # Filtramos las detecciones para cada zona
            in_zone_indices = zone_in.trigger(detections=detections)
            out_zone_indices = zone_out.trigger(detections=detections)
            
            detections_in_zone = detections[in_zone_indices]
            detections_out_zone = detections[out_zone_indices]
            
            # IMPORTANTE: Aseguramos que las propiedades se propaguen correctamente
            if hasattr(detections, 'vehicle_type') and len(in_zone_indices) > 0:
                detections_in_zone.vehicle_type = detections.vehicle_type[in_zone_indices]
            if hasattr(detections, 'vehicle_color') and len(in_zone_indices) > 0:
                detections_in_zone.vehicle_color = detections.vehicle_color[in_zone_indices]
                
            if hasattr(detections, 'vehicle_type') and len(out_zone_indices) > 0:
                detections_out_zone.vehicle_type = detections.vehicle_type[out_zone_indices]
            if hasattr(detections, 'vehicle_color') and len(out_zone_indices) > 0:
                detections_out_zone.vehicle_color = detections.vehicle_color[out_zone_indices]
            
            detections_in_zones.append(detections_in_zone)
            detections_out_zones.append(detections_out_zone)
        
        # Actualiza el gestor de detecciones con el frame_time
        detections = self.detections_manager.update(
            detections, detections_in_zones, detections_out_zones, frame_time
        )
        
        return self.annotate_frame(frame, detections)
    
    def detect_vehicle_color(self, vehicle_roi):
        """
        Detecta el color predominante de un vehículo.
        
        Args:
            vehicle_roi: Región de interés del vehículo
            
        Returns:
            str: Nombre del color predominante
        """
        # Convertir a HSV para mejor detección de color
        hsv = cv2.cvtColor(vehicle_roi, cv2.COLOR_BGR2HSV)
        
        # Definir rangos de color en HSV
        color_ranges = {
            "Rojo": [
                ((0, 100, 100), (10, 255, 255)),
                ((170, 100, 100), (180, 255, 255))
            ],
            "Naranja": [((11, 100, 100), (25, 255, 255))],
            "Amarillo": [((26, 100, 100), (35, 255, 255))],
            "Verde": [((36, 100, 100), (70, 255, 255))],
            "Azul": [((100, 100, 100), (130, 255, 255))],
            "Violeta": [((131, 100, 100), (160, 255, 255))],
            "Rosa": [((161, 100, 100), (169, 255, 255))],
            "Blanco": [((0, 0, 200), (180, 30, 255))],
            "Negro": [((0, 0, 0), (180, 255, 30))],
            "Gris": [((0, 0, 31), (180, 30, 199))],
        }
        
        # Contar píxeles para cada color
        color_counts = {}
        for color_name, ranges in color_ranges.items():
            count = 0
            for lower, upper in ranges:
                mask = cv2.inRange(hsv, lower, upper)
                count += cv2.countNonZero(mask)
            color_counts[color_name] = count
        
        # Determinar el color predominante
        total_pixels = vehicle_roi.shape[0] * vehicle_roi.shape[1]
        max_color = max(color_counts.items(), key=lambda x: x[1])
        
        # Si el color predominante representa al menos el 15% de los píxeles
        if max_color[1] > 0.15 * total_pixels:
            return max_color[0]
        else:
            return "Desconocido"


# Función para probar el procesador automático de videos
def test_automatic_processor(video_index_path, model_weights_path, output_dir="./resultados"):
    """
    Prueba el procesador automático de videos.
    
    Args:
        video_index_path: Ruta al archivo JSON con el índice de videos estables
        model_weights_path: Ruta al archivo de pesos del modelo YOLO
        output_dir: Directorio donde se guardarán los resultados
    """
    processor = AutomaticVideoProcessor(video_index_path, output_dir)
    processor.process_all_videos(model_weights_path)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        test_automatic_processor(sys.argv[1], sys.argv[2])
    else:
        print("Uso: python video_processor.py <ruta_indice_videos> <ruta_modelo_yolo>")
