import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import supervision as sv
from ultralytics import YOLO
import cv2
from typing import Dict, Iterable, List, Set
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import gc
import csv
import sys
import os
sys.path.append("Externo")  # Añadir la carpeta Externo al path
from Externo.ExtraerCordenadas import seleccionar_zonas
import json

COLORS = sv.ColorPalette.from_hex(["#E6194B", "#3CB44B", "#FFE119", "#3C76D1"])

# Variables globales para las zonas
ZONE_IN_POLYGONS = [
    np.array([[524, 681], [659, 681], [659, 846], [524, 846]]),
    np.array([[1297, 636], [1369, 636], [1369, 835], [1297, 835]]),
    np.array([[1338, 149], [1432, 149], [1432, 253], [1338, 253]]),
    np.array([[287, 52], [449, 52], [449, 206], [287, 206]]),
]

ZONE_OUT_POLYGONS = [
    np.array([[467, 403], [602, 403], [602, 552], [467, 552]]),
    np.array([[1174, 821], [1264, 821], [1264, 1006], [1174, 1006]]),
    np.array([[1331, 323], [1423, 323], [1423, 449], [1331, 449]]),
    np.array([[786, 13], [859, 13], [859, 109], [786, 109]]),
]


class ZoneManager:
    def __init__(self):
        self.config_dir = "zone_configs"
        os.makedirs(self.config_dir, exist_ok=True)

    def save_zones(self, video_name, zones_in, zones_out):
        """Guarda las zonas para un video específico en un archivo JSON"""
        config = {
            "zones_in": [z.tolist() for z in zones_in],
            "zones_out": [z.tolist() for z in zones_out]
        }

        # Usar nombre de archivo sin extensión como identificador
        base_name = os.path.splitext(os.path.basename(video_name))[0]
        file_path = os.path.join(self.config_dir, f"{base_name}_zones.json")

        with open(file_path, 'w') as f:
            json.dump(config, f, indent=2)

        return file_path

    def load_zones(self, video_name):
        """Carga zonas guardadas para un video específico"""
        base_name = os.path.splitext(os.path.basename(video_name))[0]
        file_path = os.path.join(self.config_dir, f"{base_name}_zones.json")

        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                config = json.load(f)

            zones_in = [np.array(z) for z in config["zones_in"]]
            zones_out = [np.array(z) for z in config["zones_out"]]
            return zones_in, zones_out, True

        return None, None, False


class DetectionsManager:
    """
    Maneja las detecciones a lo largo de los frames, realiza el tracking y registra
    cada evento (entrada y salida) con su información.
    """

    def __init__(self, num_rotonda: str, horario: str, dia: str, fps: float) -> None:
        self.num_rotonda = num_rotonda
        self.horario = horario
        self.dia = dia
        self.fps = fps

        self.tracker_id_to_zone_id: Dict[int, int] = {}
        # Estructura: counts[zone_out_id][zone_in_id] = set(tracker_ids)
        self.counts: Dict[int, Dict[int, Set[int]]] = {}
        # Almacena información de entrada para cada tracker_id:
        # tracker_entry_info[tracker_id] = (zone_in_id, tiempo_entrada, tipo_vehiculo)
        self.tracker_entry_info: Dict[int, tuple] = {}
        # Lista de eventos para el CSV
        self.events: List[Dict] = []
        self.event_counter = 0

    def update(
            self,
            detections_all: sv.Detections,
            detections_in_zones: List[sv.Detections],
            detections_out_zones: List[sv.Detections],
            frame_time: float
    ) -> sv.Detections:
        time_seconds = frame_time / self.fps
        """
        Actualiza el gestor de detecciones, registra la zona de entrada y, cuando un objeto
        sale, registra el evento con la información completa.
        """
        # Registro de zonas de entrada y almacenamiento de tiempo de entrada y tipo de vehículo
        for zone_in_id, detections_in_zone in enumerate(detections_in_zones):
            for idx, tracker_id in enumerate(detections_in_zone.tracker_id):
                if tracker_id not in self.tracker_entry_info:
                    try:
                        vehicle_type = detections_in_zone.vehicle_type[idx]
                    except AttributeError:
                        vehicle_type = "Desconocido"
                    # Guardar el tiempo en segundos
                    self.tracker_entry_info[tracker_id] = (zone_in_id, time_seconds, vehicle_type)
                    self.tracker_id_to_zone_id.setdefault(tracker_id, zone_in_id)

        # Registro de transición
        for zone_out_id, detections_out_zone in enumerate(detections_out_zones):
            for tracker_id in detections_out_zone.tracker_id:
                if tracker_id in self.tracker_entry_info:
                    zone_in_id, entry_time, vehicle_type = self.tracker_entry_info[tracker_id]
                    # Tiempo de salida en segundos
                    exit_time = time_seconds
                    # Tiempo dentro también en segundos
                    time_inside = exit_time - entry_time

                    # Registro en la estructura de conteos (similar a lo anterior)
                    self.counts.setdefault(zone_out_id, {})
                    self.counts[zone_out_id].setdefault(zone_in_id, set())
                    self.counts[zone_out_id][zone_in_id].add(tracker_id)

                    # Registro del evento con toda la información solicitada
                    event = {
                        "Id": tracker_id,
                        "Num rotonda": self.num_rotonda,
                        "horario": self.horario,
                        "dia": self.dia,
                        "Id_Entrada": zone_in_id,
                        "Id_salida": zone_out_id,
                        "Tiempo Entrada": entry_time,
                        "Tiempo Salida": exit_time,
                        "Tiempo dentro": time_inside,
                        "Tipo vehículo": vehicle_type
                    }
                    self.events.append(event)
                    self.event_counter += 1

                    # Una vez registrado el evento, se puede eliminar el tracker de la info de entrada
                    del self.tracker_entry_info[tracker_id]

        # Ajustamos el class_id para que coincida con la zona de entrada (para anotaciones)
        if len(detections_all) > 0:
            detections_all.class_id = np.vectorize(
                lambda x: self.tracker_id_to_zone_id.get(x, -1)
            )(detections_all.tracker_id)
        else:
            detections_all.class_id = np.array([], dtype=int)

        return detections_all[detections_all.class_id != -1]


def initiate_polygon_zones(
        polygons: List[np.ndarray],
        triggering_anchors: Iterable[sv.Position] = [sv.Position.CENTER],
) -> List[sv.PolygonZone]:
    """Crea las zonas poligonales para detecciones (in o out)."""
    return [
        sv.PolygonZone(
            polygon=polygon,
            triggering_anchors=triggering_anchors,
        )
        for polygon in polygons
    ]


class VideoProcessor:
    """
    Procesa un video con el modelo YOLO, rastrea objetos y registra eventos de cruce.
    """

    def __init__(
            self,
            source_weights_path: str,
            source_video_path: str,
            target_video_path: str,
            confidence_threshold: float,
            iou_threshold: float,
            num_rotonda: str,
            horario: str,
            dia: str,
    ) -> None:
        self.conf_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.source_video_path = source_video_path
        self.target_video_path = target_video_path

        self.model = YOLO(source_weights_path)
        self.tracker = sv.ByteTrack()
        self.tracker_id_to_vehicle_type = {}

        self.video_info = sv.VideoInfo.from_video_path(source_video_path)
        # Extraer los FPS del video para conversiones
        self.fps = self.video_info.fps
        self.zones_in = initiate_polygon_zones(ZONE_IN_POLYGONS, [sv.Position.CENTER])
        self.zones_out = initiate_polygon_zones(ZONE_OUT_POLYGONS, [sv.Position.CENTER])

        self.box_annotator = sv.BoxAnnotator(color=COLORS)
        self.label_annotator = sv.LabelAnnotator(
            color=COLORS, text_color=sv.Color.BLACK
        )
        self.trace_annotator = sv.TraceAnnotator(
            color=COLORS, position=sv.Position.CENTER, trace_length=100, thickness=2
        )
        # Se pasa el contexto (num rotonda, horario, día) al gestor de detecciones
        self.detections_manager = DetectionsManager(num_rotonda, horario, dia, self.fps)

    def process_video(self):
        """
        Procesa el video y escribe el video anotado o lo muestra en ventana.
        Al finalizar, genera un CSV con los eventos registrados.
        """
        frame_generator = sv.get_video_frames_generator(
            source_path=self.source_video_path
        )

        if self.target_video_path:
            with sv.VideoSink(self.target_video_path, self.video_info) as sink:
                with ThreadPoolExecutor(max_workers=4) as executor:
                    for frame_num, frame in enumerate(tqdm(frame_generator, total=self.video_info.total_frames)):
                        if frame_num % 2 != 0: #Reducir frames para alivianar el jale
                            continue
                        annotated_frame = executor.submit(self.process_frame, frame, frame_num).result()
                        sink.write_frame(annotated_frame)
                        gc.collect()
        else:
            for frame_num, frame in enumerate(tqdm(frame_generator, total=self.video_info.total_frames)):
                if frame_num % 2 != 0:
                    continue
                annotated_frame = self.process_frame(frame, frame_num)
                cv2.imshow("Processed Video", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            cv2.destroyAllWindows()

        self.generate_csv()

    def generate_csv(self):
        """
        Genera un archivo CSV con la siguiente estructura:
        Id, Num rotonda, horario, dia, Id_Entrada, Id_salida, Tiempo Entrada, Tiempo Salida, Tiempo dentro, Tipo vehículo
        """
        with open("informacion_importante.csv", mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([
                "Id", "Num rotonda", "horario", "dia", "Id_Entrada", "Id_salida",
                "Tiempo Entrada", "Tiempo Salida", "Tiempo dentro", "Tipo vehículo"
            ])
            for event in self.detections_manager.events:
                writer.writerow([
                    event["Id"],
                    event["Num rotonda"],
                    event["horario"],
                    event["dia"],
                    event["Id_Entrada"],
                    event["Id_salida"],
                    event["Tiempo Entrada"],
                    event["Tiempo Salida"],
                    event["Tiempo dentro"],
                    event["Tipo vehículo"],
                ])

    def annotate_frame(
            self, frame: np.ndarray, detections: sv.Detections
    ) -> np.ndarray:
        annotated_frame = frame.copy()

        # 1. Dibujar los polígonos de entrada y salida
        for i, (zone_in, zone_out) in enumerate(zip(self.zones_in, self.zones_out)):
            annotated_frame = sv.draw_polygon(
                annotated_frame, zone_in.polygon, COLORS.colors[i]
            )
            annotated_frame = sv.draw_polygon(
                annotated_frame, zone_out.polygon, COLORS.colors[i]
            )

        # 2. Construir etiquetas con ID y tipo de vehículo (opcional)
        labels = []
        for i, tracker_id in enumerate(detections.tracker_id):
            # Usar directamente el tipo guardado en el diccionario de tracking
            # que es el que guarda los valores originales del modelo
            vehicle_type = self.tracker_id_to_vehicle_type.get(tracker_id, "Desconocido")
            labels.append(f"ID:{tracker_id} - {vehicle_type}")

        # 3. Dibujar trazas, cajas y etiquetas
        annotated_frame = self.trace_annotator.annotate(annotated_frame, detections)
        annotated_frame = self.box_annotator.annotate(annotated_frame, detections)
        annotated_frame = self.label_annotator.annotate(
            annotated_frame, detections, labels
        )

        # 4. Mostrar el conteo en cada zona de salida
        for zone_out_id, zone_out in enumerate(self.zones_out):
            zone_center = sv.get_polygon_center(polygon=zone_out.polygon)
            if zone_out_id in self.detections_manager.counts:
                counts = self.detections_manager.counts[zone_out_id]
                # Por cada zona de entrada que desemboque en zone_out_id
                for j, zone_in_id in enumerate(counts):
                    count = len(counts[zone_in_id])
                    text_anchor = sv.Point(x=zone_center.x, y=zone_center.y + 40 * j)
                    annotated_frame = sv.draw_text(
                        scene=annotated_frame,
                        text=str(count),
                        text_anchor=text_anchor,
                        background_color=COLORS.colors[zone_in_id],
                    )

        return annotated_frame

    def process_frame(self, frame: np.ndarray, frame_time: float) -> np.ndarray:
        """Procesa un frame utilizando YOLO, actualiza el tracker y asigna el tipo de vehículo."""
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

        # Actualiza el tracker
        detections = self.tracker.update_with_detections(detections)

        # Actualiza el mapeo de tracker_id a tipo de vehículo
        for i, tracker_id in enumerate(detections.tracker_id):
            if i < len(detections.xyxy):
                box_key = tuple(map(float, detections.xyxy[i]))
                # Intenta encontrar el tipo de vehículo original más cercano
                if box_key in orig_vehicle_types:
                    self.tracker_id_to_vehicle_type[tracker_id] = orig_vehicle_types[box_key]

        # Asigna los tipos de vehículo usando nuestro diccionario
        vehicle_types = []
        for tracker_id in detections.tracker_id:
            vehicle_types.append(self.tracker_id_to_vehicle_type.get(tracker_id, "Desconocido"))

        detections.vehicle_type = np.array(vehicle_types)

        # Procesa las detecciones en las zonas de entrada y salida
        detections_in_zones = []
        detections_out_zones = []
        for zone_in, zone_out in zip(self.zones_in, self.zones_out):
            # Filtramos las detecciones para cada zona
            in_zone_indices = zone_in.trigger(detections=detections)
            out_zone_indices = zone_out.trigger(detections=detections)

            detections_in_zone = detections[in_zone_indices]
            detections_out_zone = detections[out_zone_indices]

            # IMPORTANTE: Aseguramos que la propiedad vehicle_type se propague correctamente
            if hasattr(detections, 'vehicle_type') and len(in_zone_indices) > 0:
                detections_in_zone.vehicle_type = detections.vehicle_type[in_zone_indices]
            if hasattr(detections, 'vehicle_type') and len(out_zone_indices) > 0:
                detections_out_zone.vehicle_type = detections.vehicle_type[out_zone_indices]

            detections_in_zones.append(detections_in_zone)
            detections_out_zones.append(detections_out_zone)

        # Actualiza el gestor de detecciones con el frame_time
        detections = self.detections_manager.update(
            detections, detections_in_zones, detections_out_zones, frame_time
        )

        return self.annotate_frame(frame, detections)


class Application(tk.Frame):
    """Interfaz Tkinter para procesar videos con la información adicional."""

    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.zone_manager = ZoneManager()
        self.pack()
        self.create_widgets()
        self.create_menu(master)

    def create_widgets(self):
        """Crea los campos y botones de la interfaz."""
        self.source_weights_path = tk.StringVar()
        self.source_video_path = tk.StringVar()
        self.target_video_path = tk.StringVar()
        self.confidence_threshold = tk.DoubleVar(value=0.3)
        self.iou_threshold = tk.DoubleVar(value=0.7)

        # Nuevos campos para el contexto
        self.num_rotonda = tk.StringVar()
        self.horario = tk.StringVar()
        self.dia = tk.StringVar()

        tk.Label(self, text="Source Weights Path").grid(row=0, column=0)
        tk.Entry(self, textvariable=self.source_weights_path).grid(row=0, column=1)
        tk.Button(self, text="Browse", command=self.browse_source_weights).grid(row=0, column=2)

        tk.Label(self, text="Source Video Path").grid(row=1, column=0)
        tk.Entry(self, textvariable=self.source_video_path).grid(row=1, column=1)
        tk.Button(self, text="Browse", command=self.browse_source_video).grid(row=1, column=2)

        tk.Label(self, text="Target Video Path").grid(row=2, column=0)
        tk.Entry(self, textvariable=self.target_video_path).grid(row=2, column=1)
        tk.Button(self, text="Browse", command=self.browse_target_video).grid(row=2, column=2)

        tk.Label(self, text="Confidence Threshold").grid(row=3, column=0)
        tk.Entry(self, textvariable=self.confidence_threshold).grid(row=3, column=1)

        tk.Label(self, text="IOU Threshold").grid(row=4, column=0)
        tk.Entry(self, textvariable=self.iou_threshold).grid(row=4, column=1)

        # Campos para la información de la rotonda
        tk.Label(self, text="Num Rotonda").grid(row=5, column=0)
        tk.Entry(self, textvariable=self.num_rotonda).grid(row=5, column=1)

        tk.Label(self, text="Horario").grid(row=6, column=0)
        tk.Entry(self, textvariable=self.horario).grid(row=6, column=1)

        tk.Label(self, text="Día").grid(row=7, column=0)
        tk.Entry(self, textvariable=self.dia).grid(row=7, column=1)

        # Botones para manejar zonas
        tk.Button(self, text="Definir zonas", command=self.define_zones).grid(row=8, column=0)
        tk.Button(self, text="Previsualizar zonas", command=self.preview_zones).grid(row=8, column=1)

        # Estado de las zonas
        self.zone_status = tk.StringVar(value="Zonas: No definidas")
        tk.Label(self, textvariable=self.zone_status).grid(row=9, column=1)

        # Botón para iniciar procesamiento
        tk.Button(self, text="Start Processing", command=self.start_processing).grid(row=10, column=1)

    def create_menu(self, master):
        """Crea el menú de la aplicación"""
        menubar = tk.Menu(master)
        master.config(menu=menubar)

        zone_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Zonas", menu=zone_menu)

        zone_menu.add_command(label="Definir zonas", command=self.define_zones)
        zone_menu.add_command(label="Previsualizar zonas", command=self.preview_zones)
        zone_menu.add_command(label="Ver configuraciones guardadas", command=self.list_all_zone_configs)

    def list_all_zone_configs(self):
        """Muestra una lista de todas las configuraciones de zonas guardadas"""
        if not os.path.exists(self.zone_manager.config_dir):
            messagebox.showinfo("Información", "No hay configuraciones guardadas")
            return

        configs = [f for f in os.listdir(self.zone_manager.config_dir) if f.endswith("_zones.json")]
        if not configs:
            messagebox.showinfo("Información", "No hay configuraciones guardadas")
            return

        config_list = "\n".join([os.path.splitext(c)[0].replace("_zones", "") for c in configs])
        messagebox.showinfo("Configuraciones guardadas", f"Videos con configuración:\n{config_list}")

    def browse_source_weights(self):
        """Selecciona el archivo de pesos YOLO."""
        file_path = filedialog.askopenfilename()
        self.source_weights_path.set(file_path)

    def browse_source_video(self):
        """Selecciona el video fuente."""
        global ZONE_IN_POLYGONS, ZONE_OUT_POLYGONS

        file_path = filedialog.askopenfilename()
        self.source_video_path.set(file_path)

        # Intenta cargar zonas automáticamente si existen
        if file_path:
            zones_in, zones_out, loaded = self.zone_manager.load_zones(file_path)
            if loaded:

                ZONE_IN_POLYGONS = zones_in
                ZONE_OUT_POLYGONS = zones_out
                self.zone_status.set(f"Zonas: {len(zones_in)} cargadas automáticamente")

    def browse_target_video(self):
        """Selecciona el video de salida."""
        file_path = filedialog.askopenfilename()
        self.target_video_path.set(file_path)
    def define_zones(self):
        """Permite al usuario definir las zonas manualmente"""
        global ZONE_IN_POLYGONS, ZONE_OUT_POLYGONS

        video_path = self.source_video_path.get()
        if not video_path:
            messagebox.showwarning("Advertencia", "Seleccione un video primero.")
            return

        # Intentar cargar zonas existentes primero
        zones_in, zones_out, loaded = self.zone_manager.load_zones(video_path)

        if loaded:
            if messagebox.askyesno("Zonas encontradas",
                            "Se encontraron zonas guardadas para este video. ¿Desea usarlas?"):
                ZONE_IN_POLYGONS = zones_in
                ZONE_OUT_POLYGONS = zones_out
                self.zone_status.set(f"Zonas: {len(zones_in)} cargadas")
                return

        # Seleccionar nuevas zonas
        zones_in, zones_out = self.select_zones_for_video(video_path)

        if zones_in is not None and zones_out is not None:
            # Guardar zonas
            self.zone_manager.save_zones(video_path, zones_in, zones_out)

            # Actualizar zonas globales

            ZONE_IN_POLYGONS = zones_in
            ZONE_OUT_POLYGONS = zones_out

            self.zone_status.set(f"Zonas: {len(zones_in)} definidas y guardadas")

    def select_zones_for_video(self, video_path):
        """Permite al usuario seleccionar zonas para un video y las guarda"""
        # Tomar un frame del video para la selección
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            messagebox.showerror("Error", "No se pudo abrir el video.")
            return None, None

        # Avanzar algunos frames para evitar frames negros al inicio
        for _ in range(30):
            ret, frame = cap.read()
            if not ret:
                break

        if not ret:
            messagebox.showerror("Error", "No se pudo leer el frame del video.")
            cap.release()
            return None, None

        # Guardar dimensiones originales
        original_height, original_width = frame.shape[:2]

        # Reducir tamaño si es necesario
        scale = min(1.0, 1200 / max(original_height, original_width))
        if scale < 1.0:
            frame = cv2.resize(frame, None, fx=scale, fy=scale)

        print("Seleccione las ZONAS DE ENTRADA (presione ESC para finalizar)")
        zones_in_polygons = seleccionar_zonas(frame, "Zonas de ENTRADA")

        print("Seleccione las ZONAS DE SALIDA (presione ESC para finalizar)")
        zones_out_polygons = seleccionar_zonas(frame, "Zonas de SALIDA")

        # Reescalar las coordenadas si fue necesario
        if scale < 1.0:
            # Escalar de vuelta a las dimensiones originales
            for zone in zones_in_polygons:
                for point in zone:
                    point[0] = int(point[0] / scale)
                    point[1] = int(point[1] / scale)

            for zone in zones_out_polygons:
                for point in zone:
                    point[0] = int(point[0] / scale)
                    point[1] = int(point[1] / scale)
        # Convertir a numpy arrays
        zones_in = np.array([np.array(poly) for poly in zones_in_polygons])
        zones_out = np.array([np.array(poly) for poly in zones_out_polygons])

        # Verificar que el número de zonas coincide
        if len(zones_in) != len(zones_out):
            messagebox.showwarning("Advertencia", "El número de zonas de entrada y salida no coincide.")

        cap.release()
        return zones_in, zones_out

    def preview_zones(self):
        """Muestra una vista previa de las zonas definidas sobre un frame del video"""
        video_path = self.source_video_path.get()
        if not video_path:
            messagebox.showwarning("Advertencia", "Seleccione un video primero.")
            return

        # Verificar si hay zonas definidas
        if len(ZONE_IN_POLYGONS) == 0 or len(ZONE_OUT_POLYGONS) == 0:
            messagebox.showwarning("Advertencia", "No hay zonas definidas. Defina las zonas primero.")
            return

        # Abrir el video y obtener un frame
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            messagebox.showerror("Error", "No se pudo abrir el video.")
            return

        # Avanzar algunos frames para evitar frames negros al inicio
        for _ in range(30):
            ret, frame = cap.read()
            if not ret:
                break

        if not ret:
            messagebox.showerror("Error", "No se pudo leer el frame del video.")
            cap.release()
            return

        # Crear una copia del frame para dibujar
        preview_frame = frame.copy()

        # Dibujar las zonas de entrada (verde) y salida (rojo)
        for polygon in ZONE_IN_POLYGONS:
            cv2.polylines(preview_frame, [polygon], True, (0, 255, 0), 2)
            # Añadir texto "Entrada" cerca del polígono
            center = np.mean(polygon, axis=0).astype(int)
            cv2.putText(preview_frame, "Entrada", tuple(center),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        for polygon in ZONE_OUT_POLYGONS:
            cv2.polylines(preview_frame, [polygon], True, (0, 0, 255), 2)
            # Añadir texto "Salida" cerca del polígono
            center = np.mean(polygon, axis=0).astype(int)
            cv2.putText(preview_frame, "Salida", tuple(center),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Redimensionar si es necesario
        height, width = preview_frame.shape[:2]
        max_display = 900
        if height > max_display or width > max_display:
            scale = min(max_display / height, max_display / width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            preview_frame = cv2.resize(preview_frame, (new_width, new_height))

        # Mostrar la vista previa
        cv2.imshow("Vista previa de zonas", preview_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cap.release()

    def start_processing(self):
        """Inicia el procesamiento del video con los parámetros indicados."""
        processor = VideoProcessor(
            source_weights_path=self.source_weights_path.get(),
            source_video_path=self.source_video_path.get(),
            target_video_path=self.target_video_path.get(),
            confidence_threshold=self.confidence_threshold.get(),
            iou_threshold=self.iou_threshold.get(),
            num_rotonda=self.num_rotonda.get(),
            horario=self.horario.get(),
            dia=self.dia.get(),
        )
        processor.process_video()
        self.master.quit()


if __name__ == "__main__":
    root = tk.Tk()
    app = Application(master=root)
    app.mainloop()
