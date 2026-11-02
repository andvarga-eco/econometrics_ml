# ------------------------------------------------------------------
# Un árbol vs. un bosque
# Simulación para la diapositiva "Un árbol vs. un bosque"
# (presentación Random Forest, Econometría con Machine Learning)
#
# Se ajusta un árbol de regresión profundo y un bosque construido
# promediando B árboles estimados en submuestras de tamaño s extraídas
# SIN reemplazo (Gaillac y L'Hour, 2025, sec. 2.7.3, ec. 2.26).
#
# Requiere solo el paquete rpart (viene con la instalación base de R).
# Ejecutar desde la carpeta "random forest": la figura se guarda en
# images/arbol_vs_bosque.png
# ------------------------------------------------------------------

library(rpart)

set.seed(3)

# ---- 1. Datos simulados -------------------------------------------
n  <- 200
mu <- function(x) sin(2 * pi * x) + 0.5 * x   # CEF verdadera
x  <- sort(runif(n))
y  <- mu(x) + rnorm(n, sd = 0.35)
datos <- data.frame(x = x, y = y)

grid <- data.frame(x = seq(0, 1, length.out = 600))

# ---- 2. Un árbol profundo ------------------------------------------
# cp = 0: sin poda; minbucket = 3: al menos 3 observaciones por hoja
ctrl <- rpart.control(cp = 0, minbucket = 3, minsplit = 6, xval = 0)

arbol      <- rpart(y ~ x, data = datos, control = ctrl)
pred_arbol <- predict(arbol, newdata = grid)

# ---- 3. Bosque: promedio de B árboles en submuestras ---------------
B <- 300                 # número de árboles
s <- floor(n / 3)        # tamaño de cada submuestra

pred_b <- matrix(NA, nrow = nrow(grid), ncol = B)
for (b in 1:B) {
  idx <- sample(n, s, replace = FALSE)            # submuestra sin reemplazo
  arbol_b <- rpart(y ~ x, data = datos[idx, ], control = ctrl)
  pred_b[, b] <- predict(arbol_b, newdata = grid)
}
pred_bosque <- rowMeans(pred_b)                   # (1/B) * sum_b T_b(x)

# ---- 4. Error cuadrático medio frente a la CEF verdadera -----------
ecm <- function(p) mean((p - mu(grid$x))^2)
cat(sprintf("ECM un árbol: %.4f\nECM bosque:   %.4f\n",
            ecm(pred_arbol), ecm(pred_bosque)))

# ---- 5. Figura -----------------------------------------------------
dir.create("images", showWarnings = FALSE)
png("images/arbol_vs_bosque.png", width = 2000, height = 920, res = 200,
    bg = "#fbfaf7")
par(mar = c(4.2, 4.2, 1, 1), bty = "l", las = 1)

plot(x, y, pch = 16, cex = 0.7, col = adjustcolor("#8a909b", 0.6),
     xlab = expression(x), ylab = expression(y))
lines(grid$x, mu(grid$x), lty = 2, lwd = 1.5, col = "#16181d")
lines(grid$x, pred_arbol, type = "s", lwd = 1.3, col = "#4a4f59")
lines(grid$x, pred_bosque, lwd = 3, col = "#d40000")

legend("bottomleft", ncol = 2, bty = "n", cex = 0.85,
       legend = c("Datos", expression(mu(x) ~ "verdadera"), "Un árbol",
                  paste0("Bosque (B = ", B, ", s = n/3)")),
       pch = c(16, NA, NA, NA), lty = c(NA, 2, 1, 1), lwd = c(NA, 1.5, 1.3, 3),
       col = c("#8a909b", "#16181d", "#4a4f59", "#d40000"))
dev.off()
