"""
Integración del contador de vehículos con el sistema de videos estables.

Este script principal integra el contador de vehículos con el sistema de videos estables,
permitiendo procesar automáticamente todos los videos estables del índice, seleccionar
visualmente las entradas y salidas, y generar un CSV con los datos de los vehículos.
"""

import os
import json
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import time

from services.drive import authenticate_drive, get_video
from services.zone_selector import ZoneSelector
from services.video_processor import AutomaticVideoProcessor, EnhancedVideoProcessor
from services.csv_generator import CSVGenerator

class VehicleCounterIntegration:
    """
    Clase principal para la integración del contador de vehículos con el sistema de videos estables.
    """
    def __init__(self, video_index_path="./videoStableIndex.json", output_dir="./resultados"):
        """
        Inicializa la integración.
        
        Args:
            video_index_path: Ruta al archivo JSON con el índice de videos estables
            output_dir: Directorio donde se guardarán los resultados
        """
        self.video_index_path = video_index_path
        self.output_dir = output_dir
        self.temp_dir = os.path.join(output_dir, "temp_videos")
        
        # Crear directorios si no existen
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Inicializar generador de CSV
        self.csv_generator = CSVGenerator(output_dir)
        self.csv_path = self.csv_generator.initialize_csv()
        
        # Autenticar en Drive
        try:
            self.drive = authenticate_drive()
        except Exception as e:
            print(f"Error al autenticar en Drive: {e}")
            self.drive = None
    
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
    
    def process_all_videos(self, model_weights_path, confidence=0.3, iou=0.7, progress_callback=None):
        """
        Procesa todos los videos estables del índice.
        
        Args:
            model_weights_path: Ruta al archivo de pesos del modelo YOLO
            confidence: Umbral de confianza para las detecciones
            iou: Umbral de IOU para las detecciones
            progress_callback: Función de callback para actualizar el progreso
            
        Returns:
            bool: True si se procesaron todos los videos correctamente
        """
        # Verificar autenticación en Drive
        if self.drive is None:
            try:
                self.drive = authenticate_drive()
            except Exception as e:
                print(f"Error al autenticar en Drive: {e}")
                return False
        
        # Cargar el índice de videos
        videos_index = self.load_video_index()
        if not videos_index:
            print("No se pudo cargar el índice de videos.")
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
                    
                    # Actualizar progreso si hay callback
                    if progress_callback:
                        progress_callback(videos_procesados, videos_totales, f"Procesando {grupo} {horario} {dia}")
                    
                    # Procesar el video
                    success = self.process_single_video(
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
        
        # Generar resumen y exportar a JSON
        self.csv_generator.generate_summary()
        self.csv_generator.export_to_json()
        
        print(f"Procesamiento completado. Se procesaron {videos_procesados} videos.")
        print(f"Resultados guardados en: {self.csv_path}")
        
        # Actualizar progreso final si hay callback
        if progress_callback:
            progress_callback(videos_totales, videos_totales, "Procesamiento completado")
        
        return True
    
    def process_single_video(self, stable_id, grupo, horario, dia, model_weights_path, confidence=0.3, iou=0.7):
        """
        Procesa un solo video estable.
        
        Args:
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
            video_path = get_video(self.drive, stable_id, self.temp_dir)
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
            
            # Crear el procesador de video mejorado
            processor = EnhancedVideoProcessor(
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
            self.csv_generator.append_events(processor.detections_manager.events, grupo, horario, dia)
            
            # 5) Limpiar archivos temporales
            if os.path.exists(video_path):
                os.remove(video_path)
            
            print(f"Video {grupo} {horario} {dia} procesado correctamente.")
            return True
            
        except Exception as e:
            print(f"Error al procesar el video {grupo} {horario} {dia}: {e}")
            return False


class IntegrationGUI:
    """
    Interfaz gráfica para la integración del contador de vehículos.
    """
    def __init__(self, master):
        """
        Inicializa la interfaz gráfica.
        
        Args:
            master: Ventana principal de Tkinter
        """
        self.master = master
        master.title("Integración del Contador de Vehículos")
        master.geometry("800x600")
        
        # Variables
        self.model_path = tk.StringVar()
        self.video_index = tk.StringVar(value="./videoStableIndex.json")
        self.output_dir = tk.StringVar(value="./resultados")
        self.confidence = tk.DoubleVar(value=0.3)
        self.iou = tk.DoubleVar(value=0.7)
        
        # Crear widgets
        self.create_widgets()
    
    def create_widgets(self):
        """Crea los widgets de la interfaz."""
        # Frame principal con padding
        main_frame = ttk.Frame(self.master, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Título
        title_label = ttk.Label(
            main_frame, 
            text="Integración del Contador de Vehículos", 
            font=("Arial", 16, "bold")
        )
        title_label.pack(pady=(0, 20))
        
        # Frame para los campos de entrada
        input_frame = ttk.LabelFrame(main_frame, text="Configuración", padding="10")
        input_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Modelo YOLO
        ttk.Label(input_frame, text="Modelo YOLO:").grid(row=0, column=0, sticky="w", pady=5)
        ttk.Entry(input_frame, textvariable=self.model_path, width=50).grid(row=0, column=1, pady=5, padx=5)
        ttk.Button(
            input_frame, 
            text="Examinar", 
            command=lambda: self.model_path.set(filedialog.askopenfilename())
        ).grid(row=0, column=2, pady=5)
        
        # Índice de videos
        ttk.Label(input_frame, text="Índice de videos:").grid(row=1, column=0, sticky="w", pady=5)
        ttk.Entry(input_frame, textvariable=self.video_index, width=50).grid(row=1, column=1, pady=5, padx=5)
        ttk.Button(
            input_frame, 
            text="Examinar", 
            command=lambda: self.video_index.set(filedialog.askopenfilename())
        ).grid(row=1, column=2, pady=5)
        
        # Directorio de salida
        ttk.Label(input_frame, text="Directorio de salida:").grid(row=2, column=0, sticky="w", pady=5)
        ttk.Entry(input_frame, textvariable=self.output_dir, width=50).grid(row=2, column=1, pady=5, padx=5)
        ttk.Button(
            input_frame, 
            text="Examinar", 
            command=lambda: self.output_dir.set(filedialog.askdirectory())
        ).grid(row=2, column=2, pady=5)
        
        # Parámetros de detección
        params_frame = ttk.LabelFrame(main_frame, text="Parámetros de detección", padding="10")
        params_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(params_frame, text="Umbral de confianza:").grid(row=0, column=0, sticky="w", pady=5)
        ttk.Scale(
            params_frame, 
            variable=self.confidence, 
            from_=0.1, 
            to=0.9, 
            orient=tk.HORIZONTAL, 
            length=200
        ).grid(row=0, column=1, pady=5, padx=5)
        ttk.Label(params_frame, textvariable=self.confidence).grid(row=0, column=2, pady=5)
        
        ttk.Label(params_frame, text="Umbral de IOU:").grid(row=1, column=0, sticky="w", pady=5)
        ttk.Scale(
            params_frame, 
            variable=self.iou, 
            from_=0.1, 
            to=0.9, 
            orient=tk.HORIZONTAL, 
            length=200
        ).grid(row=1, column=1, pady=5, padx=5)
        ttk.Label(params_frame, textvariable=self.iou).grid(row=1, column=2, pady=5)
        
        # Frame para el progreso
        progress_frame = ttk.LabelFrame(main_frame, text="Progreso", padding="10")
        progress_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            progress_frame, 
            variable=self.progress_var, 
            maximum=100, 
            length=700
        )
        self.progress_bar.pack(pady=10, fill=tk.X)
        
        self.status_var = tk.StringVar(value="Listo para iniciar")
        status_label = ttk.Label(progress_frame, textvariable=self.status_var)
        status_label.pack(pady=5)
        
        # Botones de acción
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=20)
        
        self.start_button = ttk.Button(
            button_frame, 
            text="Iniciar Procesamiento", 
            command=self.start_processing,
            width=25
        )
        self.start_button.grid(row=0, column=0, padx=10)
        
        ttk.Button(
            button_frame, 
            text="Salir", 
            command=self.master.destroy,
            width=25
        ).grid(row=0, column=1, padx=10)
    
    def update_progress(self, current, total, status):
        """
        Actualiza la barra de progreso y el estado.
        
        Args:
            current: Valor actual
            total: Valor total
            status: Estado actual
        """
        progress = (current / total) * 100 if total > 0 else 0
        self.progress_var.set(progress)
        self.status_var.set(status)
        self.master.update_idletasks()
    
    def start_processing(self):
        """Inicia el procesamiento de videos."""
        # Validar campos
        if not self.model_path.get():
            messagebox.showwarning("Advertencia", "Seleccione la ruta del modelo YOLO.")
            return
        
        if not os.path.exists(self.video_index.get()):
            messagebox.showwarning("Advertencia", "El archivo de índice de videos no existe.")
            return
        
        # Deshabilitar botón de inicio
        self.start_button.config(state=tk.DISABLED)
        
        # Crear la integración
        integration = VehicleCounterIntegration(
            video_index_path=self.video_index.get(),
            output_dir=self.output_dir.get()
        )
        
        # Iniciar procesamiento en un hilo separado
        threading.Thread(
            target=self.run_processing,
            args=(integration,),
            daemon=True
        ).start()
    
    def run_processing(self, integration):
        """
        Ejecuta el procesamiento en un hilo separado.
        
        Args:
            integration: Objeto VehicleCounterIntegration
        """
        try:
            # Procesar todos los videos
            success = integration.process_all_videos(
                model_weights_path=self.model_path.get(),
                confidence=self.confidence.get(),
                iou=self.iou.get(),
                progress_callback=self.update_progress
            )
            
            # Mostrar resultado
            if success:
                messagebox.showinfo(
                    "Procesamiento completado", 
                    f"Se han procesado todos los videos correctamente.\n\nResultados guardados en: {integration.csv_path}"
                )
            else:
                messagebox.showerror(
                    "Error", 
                    "Ha ocurrido un error durante el procesamiento. Consulte la consola para más detalles."
                )
        except Exception as e:
            messagebox.showerror("Error", f"Error inesperado: {e}")
        finally:
            # Habilitar botón de inicio
            self.start_button.config(state=tk.NORMAL)


def main():
    """Función principal para ejecutar la integración."""
    root = tk.Tk()
    app = IntegrationGUI(root)
    root.mainloop()


if __name__ == "__main__":
    # Importar las variables globales de zonas
    from services.javat import ZONE_IN_POLYGONS, ZONE_OUT_POLYGONS
    main()
