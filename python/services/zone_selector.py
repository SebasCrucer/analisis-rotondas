"""
Módulo para la selección visual de zonas de entrada y salida en videos.

Este módulo implementa una interfaz gráfica que permite al usuario seleccionar
visualmente las zonas de entrada y salida para cada video antes de procesarlo.
"""

import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox
import sys
sys.path.append("Externo")  # Añadir la carpeta Externo al path
from coordenadas import seleccionar_zonas

class ZoneSelector:
    """
    Clase para la selección visual de zonas de entrada y salida en videos.
    """
    def __init__(self, video_path, config_dir="zone_configs"):
        """
        Inicializa el selector de zonas.
        
        Args:
            video_path: Ruta al video para seleccionar zonas
            config_dir: Directorio donde se guardarán las configuraciones de zonas
        """
        self.video_path = video_path
        self.config_dir = config_dir
        self.zones_in = None
        self.zones_out = None
        
        # Crear directorio de configuraciones si no existe
        os.makedirs(config_dir, exist_ok=True)
    
    def get_config_path(self):
        """Obtiene la ruta del archivo de configuración para el video actual."""
        base_name = os.path.splitext(os.path.basename(self.video_path))[0]
        return os.path.join(self.config_dir, f"{base_name}_zones.json")
    
    def load_zones(self):
        """
        Carga las zonas guardadas para el video actual.
        
        Returns:
            tuple: (zones_in, zones_out, loaded) donde loaded es True si se cargaron zonas
        """
        import json
        config_path = self.get_config_path()
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                self.zones_in = [np.array(z) for z in config["zones_in"]]
                self.zones_out = [np.array(z) for z in config["zones_out"]]
                return self.zones_in, self.zones_out, True
            except Exception as e:
                print(f"Error al cargar zonas: {e}")
        
        return None, None, False
    
    def save_zones(self, zones_in, zones_out):
        """
        Guarda las zonas para el video actual.
        
        Args:
            zones_in: Lista de arrays numpy con las zonas de entrada
            zones_out: Lista de arrays numpy con las zonas de salida
            
        Returns:
            bool: True si se guardaron correctamente
        """
        import json
        config_path = self.get_config_path()
        
        try:
            config = {
                "zones_in": [z.tolist() for z in zones_in],
                "zones_out": [z.tolist() for z in zones_out]
            }
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            self.zones_in = zones_in
            self.zones_out = zones_out
            return True
        except Exception as e:
            print(f"Error al guardar zonas: {e}")
            return False
    
    def select_zones(self):
        """
        Abre una interfaz para que el usuario seleccione las zonas de entrada y salida.
        
        Returns:
            bool: True si se seleccionaron y guardaron zonas correctamente
        """
        # Verificar si ya existen zonas para este video
        zones_in, zones_out, loaded = self.load_zones()
        if loaded:
            # Preguntar si se quieren usar las zonas existentes
            root = tk.Tk()
            root.withdraw()  # Ocultar ventana principal
            use_existing = messagebox.askyesno(
                "Zonas encontradas",
                f"Se encontraron zonas guardadas para {os.path.basename(self.video_path)}. ¿Desea usarlas?"
            )
            root.destroy()
            
            if use_existing:
                return True
        
        # Abrir el video
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: No se pudo abrir el video {self.video_path}")
            return False
        
        # Avanzar algunos frames para evitar frames negros al inicio
        for _ in range(30):
            ret, frame = cap.read()
            if not ret:
                break
        
        if not ret:
            print(f"Error: No se pudo leer el frame del video {self.video_path}")
            cap.release()
            return False
        
        # Guardar dimensiones originales
        original_height, original_width = frame.shape[:2]
        
        # Reducir tamaño si es necesario
        scale = min(1.0, 1200 / max(original_height, original_width))
        if scale < 1.0:
            frame = cv2.resize(frame, None, fx=scale, fy=scale)
        
        # Seleccionar zonas de entrada
        print("Seleccione las ZONAS DE ENTRADA (presione ESC para finalizar)")
        zones_in_polygons = seleccionar_zonas(frame, "Zonas de ENTRADA")
        
        # Seleccionar zonas de salida
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
        zones_in = [np.array(poly) for poly in zones_in_polygons]
        zones_out = [np.array(poly) for poly in zones_out_polygons]
        
        # Verificar que hay zonas seleccionadas
        if not zones_in or not zones_out:
            print("Error: No se seleccionaron zonas")
            cap.release()
            return False
        
        # Verificar que el número de zonas coincide
        if len(zones_in) != len(zones_out):
            print(f"Advertencia: El número de zonas de entrada ({len(zones_in)}) y salida ({len(zones_out)}) no coincide")
            
            # Preguntar si se quiere continuar
            root = tk.Tk()
            root.withdraw()  # Ocultar ventana principal
            continue_anyway = messagebox.askyesno(
                "Número de zonas diferente",
                f"El número de zonas de entrada ({len(zones_in)}) y salida ({len(zones_out)}) no coincide. ¿Desea continuar de todos modos?"
            )
            root.destroy()
            
            if not continue_anyway:
                cap.release()
                return False
        
        # Guardar zonas
        success = self.save_zones(zones_in, zones_out)
        cap.release()
        
        return success
    
    def preview_zones(self):
        """
        Muestra una vista previa de las zonas seleccionadas.
        
        Returns:
            bool: True si se mostraron zonas correctamente
        """
        # Verificar si hay zonas definidas
        if self.zones_in is None or self.zones_out is None:
            zones_in, zones_out, loaded = self.load_zones()
            if not loaded:
                print("Error: No hay zonas definidas para este video")
                return False
        else:
            zones_in, zones_out = self.zones_in, self.zones_out
        
        # Abrir el video
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: No se pudo abrir el video {self.video_path}")
            return False
        
        # Avanzar algunos frames para evitar frames negros al inicio
        for _ in range(30):
            ret, frame = cap.read()
            if not ret:
                break
        
        if not ret:
            print(f"Error: No se pudo leer el frame del video {self.video_path}")
            cap.release()
            return False
        
        # Crear una copia del frame para dibujar
        preview_frame = frame.copy()
        
        # Dibujar las zonas de entrada (verde) y salida (rojo)
        for polygon in zones_in:
            cv2.polylines(preview_frame, [polygon], True, (0, 255, 0), 2)
            # Añadir texto "Entrada" cerca del polígono
            center = np.mean(polygon, axis=0).astype(int)
            cv2.putText(preview_frame, "Entrada", tuple(center),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        for polygon in zones_out:
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
        window_name = f"Vista previa de zonas - {os.path.basename(self.video_path)}"
        cv2.imshow(window_name, preview_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cap.release()
        
        return True


class ZoneSelectorApp:
    """
    Aplicación para seleccionar zonas de entrada y salida para un video.
    """
    def __init__(self, video_path):
        """
        Inicializa la aplicación de selección de zonas.
        
        Args:
            video_path: Ruta al video para seleccionar zonas
        """
        self.video_path = video_path
        self.zone_selector = ZoneSelector(video_path)
        
        # Crear ventana
        self.root = tk.Tk()
        self.root.title(f"Selección de zonas - {os.path.basename(video_path)}")
        
        # Crear widgets
        self.create_widgets()
        
        # Iniciar la aplicación
        self.root.mainloop()
    
    def create_widgets(self):
        """Crea los widgets de la interfaz."""
        frame = tk.Frame(self.root, padx=20, pady=20)
        frame.pack()
        
        # Información del video
        tk.Label(
            frame, 
            text=f"Video: {os.path.basename(self.video_path)}", 
            font=("Arial", 12, "bold")
        ).pack(pady=10)
        
        # Imagen de muestra
        self.sample_frame = self.get_sample_frame()
        if self.sample_frame is not None:
            # Redimensionar para mostrar en la interfaz
            height, width = self.sample_frame.shape[:2]
            max_display = 400
            if height > max_display or width > max_display:
                scale = min(max_display / height, max_display / width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                sample_display = cv2.resize(self.sample_frame, (new_width, new_height))
                
                # Convertir a formato para tkinter
                sample_display = cv2.cvtColor(sample_display, cv2.COLOR_BGR2RGB)
                from PIL import Image, ImageTk
                img = Image.fromarray(sample_display)
                img_tk = ImageTk.PhotoImage(image=img)
                
                # Mostrar imagen
                img_label = tk.Label(frame, image=img_tk)
                img_label.image = img_tk  # Mantener referencia
                img_label.pack(pady=10)
        
        # Botones para definir y previsualizar zonas
        button_frame = tk.Frame(frame)
        button_frame.pack(pady=10)
        
        tk.Button(
            button_frame, 
            text="Definir zonas", 
            command=self.define_zones,
            bg="#4CAF50", 
            fg="white", 
            font=("Arial", 11),
            padx=10, 
            pady=5
        ).grid(row=0, column=0, padx=10)
        
        tk.Button(
            button_frame, 
            text="Previsualizar zonas", 
            command=self.preview_zones,
            bg="#2196F3", 
            fg="white", 
            font=("Arial", 11),
            padx=10, 
            pady=5
        ).grid(row=0, column=1, padx=10)
        
        # Estado de las zonas
        self.zone_status = tk.StringVar(value="Estado: No hay zonas definidas")
        self.update_zone_status()
        tk.Label(
            frame, 
            textvariable=self.zone_status,
            font=("Arial", 11)
        ).pack(pady=10)
        
        # Botón para finalizar
        tk.Button(
            frame, 
            text="Finalizar", 
            command=self.root.destroy,
            bg="#F44336", 
            fg="white", 
            font=("Arial", 11, "bold"),
            padx=20, 
            pady=5
        ).pack(pady=10)
    
    def get_sample_frame(self):
        """Obtiene un frame de muestra del video."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return None
        
        # Avanzar algunos frames para evitar frames negros al inicio
        for _ in range(30):
            ret, frame = cap.read()
            if not ret:
                break
        
        cap.release()
        return frame if ret else None
    
    def update_zone_status(self):
        """Actualiza el estado de las zonas."""
        _, _, loaded = self.zone_selector.load_zones()
        if loaded:
            self.zone_status.set(f"Estado: Zonas definidas y guardadas")
        else:
            self.zone_status.set(f"Estado: No hay zonas definidas")
    
    def define_zones(self):
        """Define las zonas para el video."""
        if self.zone_selector.select_zones():
            self.update_zone_status()
    
    def preview_zones(self):
        """Muestra una vista previa de las zonas definidas."""
        self.zone_selector.preview_zones()


# Función para probar el selector de zonas
def test_zone_selector(video_path):
    """
    Prueba el selector de zonas con un video específico.
    
    Args:
        video_path: Ruta al video para seleccionar zonas
    """
    app = ZoneSelectorApp(video_path)


if __name__ == "__main__":
    # Si se ejecuta directamente, probar con un video de ejemplo
    if len(sys.argv) > 1:
        test_zone_selector(sys.argv[1])
    else:
        print("Uso: python zone_selector.py <ruta_al_video>")
