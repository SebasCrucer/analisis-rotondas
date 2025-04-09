library(readr)       
library(moments)    
library(corrplot)    
library(igraph)      
library(ggplot2)     

data <- read_csv("merged.csv")

data_numeric <- data[, sapply(data, is.numeric)]

summary(data_numeric)

mode_function <- function(x) {
  unique_x <- unique(x)
  unique_x[which.max(tabulate(match(x, unique_x)))]
}

moda <- sapply(data_numeric, mode_function)
curtosis <- sapply(data_numeric, kurtosis)
asimetria <- sapply(data_numeric, skewness)

estadisticos <- data.frame(Moda = moda, Curtosis = curtosis, Asimetría = asimetria)
print(estadisticos)

cuasi_gaussianas <- names(data_numeric)[abs(asimetria) < 1 & curtosis < 3]
print(paste("Variables cuasi-gaussianas:", paste(cuasi_gaussianas, collapse=", ")))

cov_matrix <- cov(data_numeric, use = "complete.obs")
cor_matrix <- cor(data_numeric, use = "complete.obs")

print("Matriz de covarianza:")
print(cov_matrix)

print("Matriz de correlación:")
print(cor_matrix)

corrplot(cor_matrix, method = "color", tl.cex = 0.2)

threshold <- 0.2
adj_matrix <- cor_matrix
adj_matrix[abs(adj_matrix) < threshold] <- 0

graph <- graph_from_adjacency_matrix(adj_matrix, mode = "directed", weighted = TRUE, diag = FALSE)

E(graph)$weight <- abs(E(graph)$weight) * 5

V(graph)$color <- ifelse(degree(graph, mode = "in") > 3, "red", "blue")

plot(graph, 
     vertex.size = 15, 
     vertex.label.cex = 0.7, 
     edge.width = E(graph)$weight, 
     edge.arrow.size = 0.5, 
     main = "Grafo de Dependencias", 
     layout = layout_with_fr)
