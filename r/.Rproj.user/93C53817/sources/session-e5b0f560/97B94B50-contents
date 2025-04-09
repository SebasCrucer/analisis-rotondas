library(dplyr)
library(tidyr)
library(ggplot2)

df <- df %>%
  mutate(
    entrada = as.factor(entrada),
    salida = as.factor(salida),
    horario = as.numeric(sub("h", "", as.character(horario)))
  )

for (r in rotondas) {
  sub_df <- df %>% filter(rotonda == r)
  for (h in 1:4) {
    df_h <- sub_df %>% filter(horario == h)
    
    frecuencia <- df_h %>%
      count(entrada, salida) %>%
      complete(entrada = factor(0:3), salida = factor(0:3), fill = list(n = 0))
    
    p <- ggplot(frecuencia, aes(x = salida, y = entrada, fill = n)) +
      geom_tile(color = "white") +
      geom_text(aes(label = n), size = 4) +
      scale_fill_gradient(low = "white", high = "red", name = "Frecuencia") +
      labs(
        title = paste("Frecuencia Entrada-Salida - Rotonda", r, "- Horario h", h, sep = ""),
        x = "Salida", y = "Entrada"
      ) +
      theme_minimal()
    
    ggsave(filename = paste0("frecuencia_maps/frecuencia_rotonda_", r, "_h", h, ".png"),
           plot = p, width = 6, height = 5)
  }
}

for (r in rotondas) {
  sub_df <- df %>% filter(rotonda == r)
  for (h in 1:4) {
    df_h <- sub_df %>% filter(horario == h)
    
    duracion <- df_h %>%
      group_by(entrada, salida) %>%
      summarise(media_duracion = mean(duracion, na.rm = TRUE), .groups = "drop") %>%
      complete(entrada = factor(0:3), salida = factor(0:3), fill = list(media_duracion = NA))
    
    duracion$entrada <- as.factor(duracion$entrada)
    duracion$salida <- as.factor(duracion$salida)
    
    p <- ggplot(duracion, aes(x = salida, y = entrada, fill = media_duracion)) +
      geom_tile(color = "white") +
      geom_text(aes(label = round(media_duracion, 1)), size = 4, na.rm = TRUE) +
      scale_fill_gradient(low = "lightgreen", high = "darkgreen", name = "Duración (seg)",
                          na.value = "grey90") +
      labs(
        title = paste("Duración Promedio - Rotonda", r, "- Horario h", h),
        x = "Salida", y = "Entrada"
      ) +
      theme_minimal()
    
    ggsave(filename = paste0("duracion_maps/duracion_rotonda_", r, "_h", h, ".png"),
           plot = p, width = 6, height = 5)
  }
}
