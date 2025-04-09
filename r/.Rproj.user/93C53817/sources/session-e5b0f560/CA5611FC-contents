library(dplyr)
library(tidyr)
library(ggplot2)

data <- read.csv("merged.csv")

calcular_vehiculos_por_min <- function(.data, grouping_vars) {
  .data %>%
    group_by(across(all_of(grouping_vars))) %>%
    summarise(
      total_veh = n(),
      t_inicio  = min(tiempo_entrada, na.rm = TRUE),
      t_fin     = max(tiempo_salida, na.rm = TRUE),
      duracion_observacion = t_fin - t_inicio,
      .groups = "drop"
    ) %>%
    mutate(
      duracion_observacion_min = duracion_observacion / 60,
      vehiculos_por_min = total_veh / duracion_observacion_min
    )
}

df_entradas <- calcular_vehiculos_por_min(data, c("rotonda", "horario", "entrada"))
ggplot(df_entradas, aes(x = horario, y = factor(entrada), fill = vehiculos_por_min)) +
  geom_tile() +
  geom_text(aes(label = round(vehiculos_por_min, 2)), color = "black") +
  facet_wrap(~ rotonda, ncol = 2) +
  labs(
    title = "Heatmap: Vehículos/min según rotonda, horario, ENTRADA",
    x = "Horario",
    y = "Entrada",
    fill = "Veh/min"
  ) +
  scale_fill_gradient(low = "white", high = "red") +
  theme_minimal()

df_salidas <- calcular_vehiculos_por_min(data, c("rotonda", "horario", "salida"))
ggplot(df_salidas, aes(x = horario, y = factor(salida), fill = vehiculos_por_min)) +
  geom_tile() +
  geom_text(aes(label = round(vehiculos_por_min, 2)), color = "black") +
  facet_wrap(~ rotonda, ncol = 2) +
  labs(
    title = "Heatmap: Vehículos/min según rotonda, horario, SALIDA",
    x = "Horario",
    y = "Salida",
    fill = "Veh/min"
  ) +
  scale_fill_gradient(low = "white", high = "red") +
  theme_minimal()

df_por_dia <- calcular_vehiculos_por_min(data, c("rotonda", "horario", "dia"))
ggplot(df_por_dia, aes(x = horario, y = dia, fill = vehiculos_por_min)) +
  geom_tile() +
  geom_text(aes(label = round(vehiculos_por_min, 2)), color = "black") +
  facet_wrap(~ rotonda, ncol = 2) +
  labs(
    title = "Heatmap: Vehículos/min según rotonda, horario y día",
    x = "Horario",
    y = "Día",
    fill = "Veh/min"
  ) +
  scale_fill_gradient(low = "white", high = "red") +
  theme_minimal()

df_por_tipo <- calcular_vehiculos_por_min(data, c("rotonda", "horario", "tipo"))
ggplot(df_por_tipo, aes(x = horario, y = tipo, fill = vehiculos_por_min)) +
  geom_tile() +
  geom_text(aes(label = round(vehiculos_por_min, 2)), color = "black") +
  facet_wrap(~ rotonda, ncol = 2) +
  labs(
    title = "Heatmap: Vehículos/min según rotonda, horario y tipo",
    x = "Horario",
    y = "Tipo de vehículo",
    fill = "Veh/min"
  ) +
  scale_fill_gradient(low = "white", high = "red") +
  theme_minimal()

df_dias <- data %>%
  group_by(rotonda, horario, dia) %>%
  summarise(
    total_veh = n(),
    t_min = min(tiempo_entrada, na.rm = TRUE),
    t_max = max(tiempo_salida, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    duracion_seg = t_max - t_min,
    duracion_min = duracion_seg / 60,
    entradas_por_min = total_veh / duracion_min,
    salidas_por_min  = total_veh / duracion_min
  )

df_mean <- df_dias %>%
  group_by(rotonda, horario) %>%
  summarise(
    mean_entradas = mean(entradas_por_min, na.rm = TRUE),
    mean_salidas  = mean(salidas_por_min,  na.rm = TRUE),
    .groups = "drop"
  )

df_long <- df_mean %>%
  pivot_longer(
    cols = c("mean_entradas", "mean_salidas"),
    names_to = "tipo",
    values_to = "valor"
  ) %>%
  mutate(
    tipo = dplyr::case_when(
      tipo == "mean_entradas" ~ "Entradas",
      tipo == "mean_salidas"  ~ "Salidas",
      TRUE ~ tipo
    )
  )

ggplot(df_long, aes(x = horario, y = rotonda, fill = valor)) +
  geom_tile() +
  geom_text(aes(label = round(valor, 2)), color = "black") +
  facet_wrap(~ tipo) +
  labs(
    title = "Media de Vehículos/min por rotonda-horario (promedio de días)",
    x = "Horario",
    y = "Rotonda",
    fill = "Veh/min"
  ) +
  scale_fill_gradient(low = "white", high = "red") +
  theme_minimal()

df_trayectos <- data %>%
  group_by(rotonda, horario, entrada, salida) %>%
  summarise(
    total_veh = n(),
    t_min = min(tiempo_entrada, na.rm = TRUE),
    t_max = max(tiempo_salida, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    duracion_seg = t_max - t_min,
    duracion_min = duracion_seg / 60,
    veh_por_min  = total_veh / duracion_min
  )

ggplot(df_trayectos, aes(x = factor(entrada), y = factor(salida), fill = veh_por_min)) +
  geom_tile() +
  geom_text(aes(label = round(veh_por_min, 2)), color = "black") +
  facet_wrap(~ rotonda + horario, ncol = 4) +
  labs(
    title = "Heatmap: Entrada/Salida (Vehículos/min) por rotonda y horario",
    x = "Entrada",
    y = "Salida",
    fill = "Veh/min"
  ) +
  scale_fill_gradient(low = "white", high = "red") +
  theme_minimal()


library(readr)
library(ggplot2)
library(reshape2)

df <- read_csv("merged.csv")

numeric_df <- df[sapply(df, is.numeric)]

crear_heatmap <- function(matrix_data, titulo, mostrar_leyenda = TRUE, decimales = 2) {
  df_melt <- melt(matrix_data)
  
  heatmap <- ggplot(df_melt, aes(Var1, Var2, fill = value)) +
    geom_tile(color = "white") +
    geom_text(aes(label = round(value, decimales)), color = "black", size = 3) +
    scale_fill_gradient2(
      low = "pink", mid = "white", high = "purple", midpoint = 0,
      guide = if (mostrar_leyenda) "colourbar" else "none"
    ) +
    labs(title = titulo, x = "", y = "") +
    theme_minimal(base_size = 12) +
    theme(
      axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1),
      plot.title = element_text(hjust = 0.5, face = "bold")
    )
  
  return(heatmap)
}

cor_matrix <- cor(numeric_df, use = "complete.obs")
cov_matrix <- cov(numeric_df, use = "complete.obs")

heatmap_cor <- crear_heatmap(cor_matrix, "Heatmap de Correlación", mostrar_leyenda = FALSE)
heatmap_cov <- crear_heatmap(cov_matrix, "Heatmap de Covarianza", mostrar_leyenda = TRUE)

print(heatmap_cor)
print(heatmap_cov)
