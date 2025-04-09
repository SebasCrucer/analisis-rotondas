"""
Módulo para la generación de CSV con datos de vehículos.

Este módulo implementa la funcionalidad para generar un archivo CSV con los datos
de los vehículos detectados en los videos procesados, incluyendo todos los campos
requeridos: Id, rotonda, horario, día, Id_Entrada, Id_salida, Tiempo Entrada,
Tiempo Salida, Tipo vehículo y Color.
"""

import os
import csv
import json
from datetime import datetime
import numpy as np

class CSVGenerator:
    """
    Clase para generar archivos CSV con datos de vehículos.
    """
    def __init__(self, output_dir="./resultados"):
        """
        Inicializa el generador de CSV.
        
        Args:
            output_dir: Directorio donde se guardarán los archivos CSV
        """
        self.output_dir = output_dir
        self.csv_path = os.path.join(output_dir, "vehiculos_contados.csv")
        
        # Crear directorio si no existe
        os.makedirs(output_dir, exist_ok=True)
    
    def initialize_csv(self):
        """
        Inicializa el archivo CSV con los encabezados requeridos.
        
        Returns:
            str: Ruta al archivo CSV creado
        """
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
        
        return self.csv_path
    
    def append_events(self, events, rotonda, horario, dia):
        """
        Añade eventos al archivo CSV.
        
        Args:
            events: Lista de eventos de vehículos detectados
            rotonda: Identificador de la rotonda (r1, r2, etc.)
            horario: Identificador del horario (h1, h2, etc.)
            dia: Identificador del día (m, t, w)
            
        Returns:
            int: Número de eventos añadidos
        """
        # Verificar si el archivo existe, si no, inicializarlo
        if not os.path.exists(self.csv_path):
            self.initialize_csv()
        
        count = 0
        with open(self.csv_path, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            for event in events:
                # Extraer el color si está disponible
                color = event.get("Color", "")
                if not color and hasattr(event, "vehicle_color"):
                    color = event.vehicle_color
                
                # Formatear tiempos como HH:MM:SS.ms
                tiempo_entrada = self.format_time(event["Tiempo Entrada"])
                tiempo_salida = self.format_time(event["Tiempo Salida"])
                
                writer.writerow([
                    event["Id"],
                    rotonda,
                    horario,
                    dia,
                    event["Id_Entrada"],
                    event["Id_salida"],
                    tiempo_entrada,
                    tiempo_salida,
                    event["Tipo vehículo"],
                    color
                ])
                count += 1
        
        return count
    
    def format_time(self, seconds):
        """
        Formatea un tiempo en segundos como HH:MM:SS.ms
        
        Args:
            seconds: Tiempo en segundos
            
        Returns:
            str: Tiempo formateado
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds_remainder = seconds % 60
        
        return f"{hours:02d}:{minutes:02d}:{seconds_remainder:.3f}"
    
    def merge_csv_files(self, csv_files, output_file=None):
        """
        Combina varios archivos CSV en uno solo.
        
        Args:
            csv_files: Lista de rutas a archivos CSV
            output_file: Ruta al archivo CSV de salida (opcional)
            
        Returns:
            str: Ruta al archivo CSV combinado
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.output_dir, f"vehiculos_combinados_{timestamp}.csv")
        
        # Inicializar el archivo de salida con los encabezados
        with open(output_file, mode="w", newline="", encoding="utf-8") as outfile:
            writer = csv.writer(outfile)
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
        
        # Combinar los archivos
        total_rows = 0
        for csv_file in csv_files:
            if os.path.exists(csv_file):
                with open(csv_file, mode="r", encoding="utf-8") as infile:
                    reader = csv.reader(infile)
                    next(reader, None)  # Saltar encabezados
                    
                    with open(output_file, mode="a", newline="", encoding="utf-8") as outfile:
                        writer = csv.writer(outfile)
                        for row in reader:
                            writer.writerow(row)
                            total_rows += 1
        
        print(f"Se combinaron {len(csv_files)} archivos CSV con un total de {total_rows} filas.")
        return output_file
    
    def generate_summary(self, output_file=None):
        """
        Genera un resumen de los datos del CSV.
        
        Args:
            output_file: Ruta al archivo de resumen (opcional)
            
        Returns:
            str: Ruta al archivo de resumen
        """
        if output_file is None:
            output_file = os.path.join(self.output_dir, "resumen_vehiculos.txt")
        
        if not os.path.exists(self.csv_path):
            print(f"Error: No se encontró el archivo CSV {self.csv_path}")
            return None
        
        # Contadores
        total_vehiculos = 0
        vehiculos_por_rotonda = {}
        vehiculos_por_horario = {}
        vehiculos_por_dia = {}
        vehiculos_por_tipo = {}
        vehiculos_por_color = {}
        
        # Leer el CSV
        with open(self.csv_path, mode="r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                total_vehiculos += 1
                
                # Contar por rotonda
                rotonda = row["rotonda (r)"]
                vehiculos_por_rotonda[rotonda] = vehiculos_por_rotonda.get(rotonda, 0) + 1
                
                # Contar por horario
                horario = row["horario (h)"]
                vehiculos_por_horario[horario] = vehiculos_por_horario.get(horario, 0) + 1
                
                # Contar por día
                dia = row["dia (m, t o w)"]
                vehiculos_por_dia[dia] = vehiculos_por_dia.get(dia, 0) + 1
                
                # Contar por tipo
                tipo = row["Tipo vehículo"]
                vehiculos_por_tipo[tipo] = vehiculos_por_tipo.get(tipo, 0) + 1
                
                # Contar por color
                color = row["Color"]
                if color:  # Solo contar si hay color
                    vehiculos_por_color[color] = vehiculos_por_color.get(color, 0) + 1
        
        # Escribir el resumen
        with open(output_file, mode="w", encoding="utf-8") as file:
            file.write("=== RESUMEN DE VEHÍCULOS DETECTADOS ===\n\n")
            file.write(f"Total de vehículos: {total_vehiculos}\n\n")
            
            file.write("--- Por rotonda ---\n")
            for rotonda, count in sorted(vehiculos_por_rotonda.items()):
                file.write(f"{rotonda}: {count} vehículos ({count/total_vehiculos*100:.1f}%)\n")
            
            file.write("\n--- Por horario ---\n")
            for horario, count in sorted(vehiculos_por_horario.items()):
                file.write(f"{horario}: {count} vehículos ({count/total_vehiculos*100:.1f}%)\n")
            
            file.write("\n--- Por día ---\n")
            for dia, count in sorted(vehiculos_por_dia.items()):
                file.write(f"{dia}: {count} vehículos ({count/total_vehiculos*100:.1f}%)\n")
            
            file.write("\n--- Por tipo de vehículo ---\n")
            for tipo, count in sorted(vehiculos_por_tipo.items(), key=lambda x: x[1], reverse=True):
                file.write(f"{tipo}: {count} vehículos ({count/total_vehiculos*100:.1f}%)\n")
            
            file.write("\n--- Por color ---\n")
            for color, count in sorted(vehiculos_por_color.items(), key=lambda x: x[1], reverse=True):
                file.write(f"{color}: {count} vehículos ({count/total_vehiculos*100:.1f}%)\n")
        
        print(f"Resumen generado en: {output_file}")
        return output_file
    
    def export_to_json(self, output_file=None):
        """
        Exporta los datos del CSV a formato JSON.
        
        Args:
            output_file: Ruta al archivo JSON de salida (opcional)
            
        Returns:
            str: Ruta al archivo JSON
        """
        if output_file is None:
            output_file = os.path.join(self.output_dir, "vehiculos_contados.json")
        
        if not os.path.exists(self.csv_path):
            print(f"Error: No se encontró el archivo CSV {self.csv_path}")
            return None
        
        # Leer el CSV y convertir a JSON
        data = []
        with open(self.csv_path, mode="r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                data.append(dict(row))
        
        # Escribir el JSON
        with open(output_file, mode="w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False)
        
        print(f"Datos exportados a JSON en: {output_file}")
        return output_file


class DetectionEventManager:
    """
    Clase para gestionar eventos de detección de vehículos y generar CSV.
    Extiende la funcionalidad de DetectionsManager en javat.py.
    """
    def __init__(self, num_rotonda, horario, dia, fps, csv_generator=None):
        """
        Inicializa el gestor de eventos de detección.
        
        Args:
            num_rotonda: Identificador de la rotonda (r1, r2, etc.)
            horario: Identificador del horario (h1, h2, etc.)
            dia: Identificador del día (m, t, w)
            fps: Frames por segundo del video
            csv_generator: Generador de CSV (opcional)
        """
        self.num_rotonda = num_rotonda
        self.horario = horario
        self.dia = dia
        self.fps = fps
        self.csv_generator = csv_generator
        
        # Estructuras para seguimiento
        self.tracker_id_to_zone_id = {}
        self.counts = {}
        self.tracker_entry_info = {}
        self.events = []
        self.event_counter = 0
        
        # Información adicional para color
        self.tracker_id_to_color = {}
    
    def set_vehicle_color(self, tracker_id, color):
        """
        Establece el color de un vehículo.
        
        Args:
            tracker_id: ID del tracker
            color: Color del vehículo
        """
        self.tracker_id_to_color[tracker_id] = color
    
    def update(self, detections_all, detections_in_zones, detections_out_zones, frame_time):
        """
        Actualiza el gestor de detecciones, registra la zona de entrada y, cuando un objeto
        sale, registra el evento con la información completa.
        
        Args:
            detections_all: Todas las detecciones
            detections_in_zones: Detecciones en zonas de entrada
            detections_out_zones: Detecciones en zonas de salida
            frame_time: Tiempo del frame
            
        Returns:
            Detecciones actualizadas
        """
        time_seconds = frame_time / self.fps
        
        # Registro de zonas de entrada y almacenamiento de tiempo de entrada y tipo de vehículo
        for zone_in_id, detections_in_zone in enumerate(detections_in_zones):
            for idx, tracker_id in enumerate(detections_in_zone.tracker_id):
                if tracker_id not in self.tracker_entry_info:
                    try:
                        vehicle_type = detections_in_zone.vehicle_type[idx]
                    except (AttributeError, IndexError):
                        vehicle_type = "Desconocido"
                    
                    # Obtener color si está disponible
                    try:
                        vehicle_color = detections_in_zone.vehicle_color[idx]
                    except (AttributeError, IndexError):
                        vehicle_color = self.tracker_id_to_color.get(tracker_id, "Desconocido")
                    
                    # Guardar el tiempo en segundos y la información del vehículo
                    self.tracker_entry_info[tracker_id] = (zone_in_id, time_seconds, vehicle_type, vehicle_color)
                    self.tracker_id_to_zone_id.setdefault(tracker_id, zone_in_id)
        
        # Registro de transición
        for zone_out_id, detections_out_zone in enumerate(detections_out_zones):
            for tracker_id in detections_out_zone.tracker_id:
                if tracker_id in self.tracker_entry_info:
                    zone_in_id, entry_time, vehicle_type, vehicle_color = self.tracker_entry_info[tracker_id]
                    # Tiempo de salida en segundos
                    exit_time = time_seconds
                    # Tiempo dentro también en segundos
                    time_inside = exit_time - entry_time
                    
                    # Registro en la estructura de conteos
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
                        "Tipo vehículo": vehicle_type,
                        "Color": vehicle_color
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
    
    def save_events_to_csv(self):
        """
        Guarda los eventos en un archivo CSV.
        
        Returns:
            int: Número de eventos guardados
        """
        if self.csv_generator is None:
            print("Error: No se ha configurado un generador de CSV")
            return 0
        
        return self.csv_generator.append_events(
            self.events, 
            self.num_rotonda, 
            self.horario, 
            self.dia
        )


# Función para probar el generador de CSV
def test_csv_generator(output_dir="./resultados"):
    """
    Prueba el generador de CSV con datos de ejemplo.
    
    Args:
        output_dir: Directorio donde se guardarán los archivos CSV
    """
    import numpy as np
    
    # Crear generador de CSV
    csv_gen = CSVGenerator(output_dir)
    csv_path = csv_gen.initialize_csv()
    
    # Crear algunos eventos de ejemplo
    events = []
    for i in range(10):
        event = {
            "Id": i + 1,
            "Num rotonda": "r1",
            "horario": "h1",
            "dia": "m",
            "Id_Entrada": np.random.randint(0, 4),
            "Id_salida": np.random.randint(0, 4),
            "Tiempo Entrada": np.random.uniform(0, 100),
            "Tiempo Salida": np.random.uniform(100, 200),
            "Tiempo dentro": np.random.uniform(10, 100),
            "Tipo vehículo": np.random.choice(["coche", "camión", "moto", "bus"]),
            "Color": np.random.choice(["Rojo", "Azul", "Negro", "Blanco", "Gris"])
        }
        events.append(event)
    
    # Añadir eventos al CSV
    csv_gen.append_events(events, "r1", "h1", "m")
    
    # Generar resumen
    csv_gen.generate_summary()
    
    # Exportar a JSON
    csv_gen.export_to_json()
    
    print(f"Prueba completada. Archivos generados en: {output_dir}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        test_csv_generator(sys.argv[1])
    else:
        test_csv_generator()
