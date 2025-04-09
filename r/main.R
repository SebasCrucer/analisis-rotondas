data <- read.csv("merged.csv")

crear_momentos <- function(nombres_variables) {
  function(lista_variables) {
    resultados <- mapply(function(x, nombre) {
      n <- length(x)
      media <- mean(x, na.rm = TRUE)
      mediana <- median(x, na.rm = TRUE)
      desv <- sd(x, na.rm = TRUE)
      asimetria <- sum((x - media)^3, na.rm = TRUE) / (n * desv^3)
      curtosis <- sum((x - media)^4, na.rm = TRUE) / (n * desv^4)
      
      resultado <- c(
        media      = media, 
        mediana    = mediana, 
        desviacion = desv,
        asimetria  = asimetria, 
        curtosis   = curtosis
      )
      return(resultado)
    }, lista_variables, nombres_variables, SIMPLIFY = FALSE)
    
    df <- as.data.frame(do.call(rbind, resultados))
    rownames(df) <- nombres_variables
    return(df)
  }
}


moments_vars_name <- c(
  "Duración global",
  "Duración en r1-h1",
  "Duración en r1-h2",
  "Duración en r1-h3",
  "Duración en r1-h4",
  "Duración en r2-h1",
  "Duración en r2-h2",
  "Duración en r2-h3",
  "Duración en r2-h4"
)

moments_vars <- list(
  data$duracion,
  subset(data, rotonda == "r1" & horario == "h1")$duracion,
  subset(data, rotonda == "r1" & horario == "h2")$duracion,
  subset(data, rotonda == "r1" & horario == "h3")$duracion,
  subset(data, rotonda == "r1" & horario == "h4")$duracion,
  subset(data, rotonda == "r2" & horario == "h1")$duracion,
  subset(data, rotonda == "r2" & horario == "h2")$duracion,
  subset(data, rotonda == "r2" & horario == "h3")$duracion,
  subset(data, rotonda == "r2" & horario == "h4")$duracion
)

moments_function <- crear_momentos(moments_vars_name)
tabla_momentos <- moments_function(moments_vars)

shapiro_p_values <- sapply(moments_vars, function(x) {
  x_no_na <- x[!is.na(x)]
  if(length(x_no_na) >= 3 && length(x_no_na) <= 5000) {
    return(shapiro.test(x_no_na)$p.value)
  } else {
    return(NA)
  }
})

umbral <- 0.05
es_gaussiana <- ifelse(shapiro_p_values > umbral, "Sí", "No")

tabla_momentos$shapiro_p_value <- shapiro_p_values
tabla_momentos$Es_Gaussiana   <- es_gaussiana

par(mfrow = c(3, 3), mar = c(3, 3, 2, 1))

for(i in seq_along(moments_vars)) {
  x_i <- moments_vars[[i]]
  x_i <- x_i[!is.na(x_i)] 
  
  var_name <- moments_vars_name[i]
  
  media_i  <- tabla_momentos$media[i]
  sd_i     <- tabla_momentos$desviacion[i]
  
  hist(x_i,
       probability = TRUE,
       breaks = 30,
       main = paste("Histograma de", var_name),
       xlab = "Duración")
  
  curve(dnorm(x, mean = media_i, sd = sd_i),
        col = "blue", lwd = 2, add = TRUE)
  
  legend("topright",
         legend = c(
           paste("p-value:", shapiro_p_values[i]),
           paste("Gaussiana?:", es_gaussiana[i])
         ),
         bty = "n")
}

par(mfrow = c(1, 1))

print(tabla_momentos)

x <- data$duracion
x <- x + 0.000001 

norm_m  <- mean(x, na.rm = TRUE)
norm_sd <- sd(x, na.rm = TRUE)

exp_rate <- 1 / mean(x, na.rm = TRUE)

log_x <- log(x)
lnorm_mean <- mean(log_x, na.rm = TRUE)
lnorm_sd   <- sd(log_x, na.rm = TRUE)

gumbel_beta <- sqrt(6 * var(x, na.rm = TRUE)) / pi
gumbel_mu   <- mean(x, na.rm = TRUE) - 0.5772 * gumbel_beta

hist(x,
     breaks = 40,
     probability = TRUE,
     col = "lightgray",
     main = "Comparación de Distribuciones (Duración Global)",
     xlab = "Duración")

curve(dnorm(x, mean = norm_m, sd = norm_sd),
      col = "blue", lwd = 2, add = TRUE)

curve(dexp(x, rate = exp_rate),
      col = "red", lwd = 2, add = TRUE)

curve(dlnorm(x, meanlog = lnorm_mean, sdlog = lnorm_sd),
      col = "darkgreen", lwd = 2, add = TRUE)

curve({
  (1/gumbel_beta) * exp(-(x - gumbel_mu)/gumbel_beta) *
    exp(-exp(-(x - gumbel_mu)/gumbel_beta))
},
col = "purple", lwd = 2, add = TRUE)

legend("topright",
       legend = c("Normal", "Exponencial", "Lognormal", "Gumbel"),
       col = c("blue", "red", "darkgreen", "purple"),
       lwd = 2)
