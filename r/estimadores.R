if (!require("dplyr")) install.packages("dplyr")
if (!require("ggplot2")) install.packages("ggplot2")

library(dplyr)
library(ggplot2)

dir.create("graficas_estimaciones", showWarnings = FALSE)

dataset <- read.csv("merged.csv")

datos_r1 <- dataset %>% filter(rotonda == "r1")
datos_r2 <- dataset %>% filter(rotonda == "r2")

graficar_mle <- function(data, columna, titulo_extra = "") {
  sensor <- data[[columna]]
  set.seed(10)
  n <- length(sensor)
  x <- 1:n
  mu <- cumsum(sensor) / x
  s <- sqrt(cumsum((sensor - mu)^2) / x)
  
  df_plot <- data.frame(
    Muestra = x,
    Valor = sensor,
    Media = mu,
    Superior = mu + s,
    Inferior = mu - s
  )
  
  ggplot(df_plot, aes(x = Muestra)) +
    geom_point(aes(y = Valor), color = "gray50", size = 1.2, alpha = 0.5) +
    geom_line(aes(y = Media), color = "red", size = 1.2) +
    geom_line(aes(y = Superior), color = "purple", linetype = "dashed") +
    geom_line(aes(y = Inferior), color = "purple", linetype = "dashed") +
    labs(title = paste("Estimación MLE -", columna, titulo_extra),
         y = "Valor", x = "Índice de muestra") +
    theme_minimal()
}

graficar_em <- function(data, columna, p = 0.7, titulo_extra = "") {
  sensor <- data[[columna]]
  set.seed(10)
  n <- length(sensor)
  mu <- numeric(n)
  mu[1] <- sensor[1]
  
  for (i in 2:n) {
    mu[i] <- p * mu[i - 1] + (1 - p) * sensor[i]
  }
  
  df_plot <- data.frame(
    Muestra = 1:n,
    Valor = sensor,
    EM = mu
  )
  
  ggplot(df_plot, aes(x = Muestra)) +
    geom_point(aes(y = Valor), color = "gray50", size = 1.2, alpha = 0.5) +
    geom_line(aes(y = EM), color = "blue", size = 1.2) +
    labs(title = paste("Estimación EM -", columna, titulo_extra),
         y = "Valor", x = "Índice de muestra") +
    theme_minimal()
}

columnas <- c("tiempo_entrada", "tiempo_salida", "duracion")

for (col in columnas) {
  g1 <- graficar_mle(datos_r1, col, " (r1)")
  ggsave(filename = paste0("graficas_estimaciones/MLE_", col, "_r1.png"),
         plot = g1, width = 8, height = 5)

  g2 <- graficar_mle(datos_r2, col, " (r2)")
  ggsave(filename = paste0("graficas_estimaciones/MLE_", col, "_r2.png"),
         plot = g2, width = 8, height = 5)

  g3 <- graficar_em(datos_r1, col, 0.7, " (r1)")
  ggsave(filename = paste0("graficas_estimaciones/EM_", col, "_r1.png"),
         plot = g3, width = 8, height = 5)

  g4 <- graficar_em(datos_r2, col, 0.7, " (r2)")
  ggsave(filename = paste0("graficas_estimaciones/EM_", col, "_r2.png"),
         plot = g4, width = 8, height = 5)
}

