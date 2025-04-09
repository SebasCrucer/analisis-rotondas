"""
Módulo para extraer coordenadas de polígonos seleccionados manualmente en un frame de video.

Este módulo permite al usuario seleccionar polígonos de forma visual en el primer frame
de un video, para definir zonas de entrada y salida para el contador de vehículos.
"""

import cv2
import numpy as np

class PolySelector:
    """
    Clase para seleccionar polígonos en una imagen.
    """
    def __init__(self, window_name):
        """
        Inicializa el selector de polígonos.
        
        Args:
            window_name: Nombre de la ventana de OpenCV
        """
        self.window_name = window_name
        self.image = None
        self.temp_image = None
        self.points = []
        self.polygons = []
        self.is_drawing = False
        self.done = False
    
    def mouse_callback(self, event, x, y, flags, param):
        """
        Callback para eventos del mouse.
        
        Args:
            event: Tipo de evento del mouse
            x, y: Coordenadas del mouse
            flags: Flags adicionales
            param: Parámetros adicionales
        """
        if self.done:
            return
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Iniciar un nuevo punto
            self.is_drawing = True
            self.points.append((x, y))
            # Dibujar el punto
            cv2.circle(self.temp_image, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow(self.window_name, self.temp_image)
        
        elif event == cv2.EVENT_MOUSEMOVE and self.is_drawing:
            # Actualizar la imagen temporal para mostrar la línea actual
            temp = self.temp_image.copy()
            if len(self.points) > 0:
                cv2.line(temp, self.points[-1], (x, y), (0, 255, 0), 2)
            cv2.imshow(self.window_name, temp)
        
        elif event == cv2.EVENT_LBUTTONUP and self.is_drawing:
            # Finalizar el punto actual
            self.is_drawing = False
            # Si hay más de un punto, dibujar la línea
            if len(self.points) > 1:
                cv2.line(self.temp_image, self.points[-2], self.points[-1], (0, 255, 0), 2)
                cv2.imshow(self.window_name, self.temp_image)
    
    def select_polygon(self, image):
        """
        Permite al usuario seleccionar un polígono en la imagen.
        
        Args:
            image: Imagen donde seleccionar el polígono
            
        Returns:
            list: Lista de puntos del polígono o None si se canceló
        """
        self.image = image.copy()
        self.temp_image = image.copy()
        self.points = []
        self.is_drawing = False
        
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        cv2.putText(self.temp_image, "Seleccione puntos del poligono. ESC para finalizar, ENTER para cerrar poligono", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow(self.window_name, self.temp_image)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            # ESC para cancelar o finalizar
            if key == 27:
                if len(self.points) == 0:
                    # Cancelar si no hay puntos
                    cv2.destroyWindow(self.window_name)
                    return None
                else:
                    # Finalizar el polígono actual
                    if len(self.points) >= 3:
                        # Cerrar el polígono
                        cv2.line(self.temp_image, self.points[-1], self.points[0], (0, 255, 0), 2)
                        cv2.imshow(self.window_name, self.temp_image)
                        polygon = self.points.copy()
                        cv2.destroyWindow(self.window_name)
                        return polygon
                    else:
                        # No hay suficientes puntos para un polígono
                        cv2.destroyWindow(self.window_name)
                        return None
            
            # ENTER para cerrar el polígono actual
            elif key == 13:
                if len(self.points) >= 3:
                    # Cerrar el polígono
                    cv2.line(self.temp_image, self.points[-1], self.points[0], (0, 255, 0), 2)
                    cv2.imshow(self.window_name, self.temp_image)
                    polygon = self.points.copy()
                    cv2.destroyWindow(self.window_name)
                    return polygon
    
    def select_multiple_polygons(self, image):
        """
        Permite al usuario seleccionar múltiples polígonos en la imagen.
        
        Args:
            image: Imagen donde seleccionar los polígonos
            
        Returns:
            list: Lista de polígonos (cada polígono es una lista de puntos)
        """
        self.image = image.copy()
        self.temp_image = image.copy()
        self.polygons = []
        self.done = False
        
        cv2.namedWindow(self.window_name)
        
        cv2.putText(self.temp_image, "Seleccione poligonos. ESC para finalizar todos, ENTER para cerrar poligono actual", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow(self.window_name, self.temp_image)
        
        while not self.done:
            # Seleccionar un polígono
            polygon = self.select_polygon(self.temp_image)
            
            if polygon is None:
                # El usuario presionó ESC sin puntos, finalizar
                self.done = True
            else:
                # Añadir el polígono a la lista
                self.polygons.append(polygon)
                
                # Dibujar el polígono en la imagen
                pts = np.array(polygon, np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(self.temp_image, [pts], True, (0, 255, 0), 2)
                
                # Mostrar el número del polígono
                centroid = np.mean(polygon, axis=0, dtype=np.int32)
                cv2.putText(self.temp_image, str(len(self.polygons)), 
                           tuple(centroid), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                cv2.putText(self.temp_image, "Poligono " + str(len(self.polygons)) + " añadido. ESC para finalizar, cualquier tecla para continuar", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow(self.window_name, self.temp_image)
                
                # Esperar a que el usuario decida si continuar o finalizar
                key = cv2.waitKey(0) & 0xFF
                if key == 27:
                    # ESC para finalizar
                    self.done = True
                else:
                    # Cualquier otra tecla para continuar
                    # Limpiar el mensaje de continuar/finalizar
                    self.temp_image = self.image.copy()
                    # Redibujar todos los polígonos
                    for i, poly in enumerate(self.polygons):
                        pts = np.array(poly, np.int32)
                        pts = pts.reshape((-1, 1, 2))
                        cv2.polylines(self.temp_image, [pts], True, (0, 255, 0), 2)
                        centroid = np.mean(poly, axis=0, dtype=np.int32)
                        cv2.putText(self.temp_image, str(i+1), 
                                   tuple(centroid), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    cv2.putText(self.temp_image, "Seleccione poligonos. ESC para finalizar todos, ENTER para cerrar poligono actual", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.imshow(self.window_name, self.temp_image)
        
        cv2.destroyAllWindows()
        return self.polygons


def seleccionar_zonas(frame, titulo="Selección de Zonas"):
    """
    Función principal para seleccionar zonas en un frame.
    
    Args:
        frame: Frame donde seleccionar las zonas
        titulo: Título de la ventana
        
    Returns:
        list: Lista de polígonos seleccionados (cada polígono es una lista de puntos)
    """
    selector = PolySelector(titulo)
    polygons = selector.select_multiple_polygons(frame)
    
    # Convertir a formato de lista de numpy arrays para compatibilidad con el resto del código
    return [np.array(polygon) for polygon in polygons]


if __name__ == "__main__":
    # Ejemplo de uso
    import sys
    
    if len(sys.argv) > 1:
        # Usar un archivo de imagen o video como entrada
        cap = cv2.VideoCapture(sys.argv[1])
        ret, frame = cap.read()
        if not ret:
            print("Error al leer el archivo")
            sys.exit(1)
        cap.release()
    else:
        # Crear una imagen en blanco
        frame = np.zeros((600, 800, 3), np.uint8)
        frame.fill(255)  # Fondo blanco
    
    print("Seleccione zonas de entrada:")
    zonas_entrada = seleccionar_zonas(frame, "Zonas de Entrada")
    
    print("Seleccione zonas de salida:")
    zonas_salida = seleccionar_zonas(frame, "Zonas de Salida")
    
    print(f"Se seleccionaron {len(zonas_entrada)} zonas de entrada y {len(zonas_salida)} zonas de salida")
    
    # Mostrar las zonas seleccionadas
    for i, zona in enumerate(zonas_entrada):
        pts = zona.reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
        centroid = np.mean(zona, axis=0, dtype=np.int32)
        cv2.putText(frame, f"E{i+1}", tuple(centroid), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    for i, zona in enumerate(zonas_salida):
        pts = zona.reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], True, (0, 0, 255), 2)
        centroid = np.mean(zona, axis=0, dtype=np.int32)
        cv2.putText(frame, f"S{i+1}", tuple(centroid), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow("Zonas Seleccionadas", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
