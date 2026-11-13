#  Banco de preguntas — Econometría con ML (202630)
#  type: "mc" (opción múltiple, ans = índice de la correcta) | "num" (ans = valor, tol = tolerancia absoluta)
#  lvl: 1 básico, 2 intermedio, 3 avanzado

TOPICS = [
    {"id": "intro", "name": "Predicción y causalidad", "deck": "Introducción: predicción, estimación y causalidad"},
    {"id": "mco", "name": "MCO y alta dimensionalidad", "deck": "MCO: alta dimensionalidad"},
    {"id": "lasso", "name": "Lasso", "deck": "LASSO: regularización y selección"},
    {"id": "causal", "name": "Inferencia causal", "deck": "Inferencia causal: resultados potenciales"},
    {"id": "dlasso", "name": "Doble Lasso", "deck": "Doble Lasso: inferencia"},
    {"id": "rf", "name": "Árboles y bosques", "deck": "Random Forest"},
    {"id": "dml", "name": "DML", "deck": "Double/debiased machine learning"},
]

Q = []
def mc(id, t, lvl, q, opts, ans, exp, code=None):
    Q.append(dict(id=id, t=t, lvl=lvl, type="mc", q=q, opts=opts, ans=ans, exp=exp, code=code))
def num(id, t, lvl, q, ans, tol, exp, code=None, unit=None):
    Q.append(dict(id=id, t=t, lvl=lvl, type="num", q=q, ans=ans, tol=tol, exp=exp, code=code, unit=unit))

# ───────────────────────── INTRO ─────────────────────────
mc("i1", "intro", 1,
   r"Ante nubes en el cielo hay que decidir si llevar sombrilla; ante una sequía, si invertir en un baile para la lluvia. ¿Cuál es un problema de <b>predicción</b>?",
   [r"La sombrilla", r"El baile para la lluvia", r"Ambos", r"Ninguno: los dos son causales"], 0,
   r"La sombrilla no cambia la probabilidad de lluvia: $\partial Y/\partial X_0=0$. El valor de la decisión depende solo de predecir bien $Y$. En el baile ocurre lo contrario: $\partial\pi/\partial X_0=0$ y todo depende del efecto causal $\partial Y/\partial X_0$.")
mc("i2", "intro", 1,
   r"En la descomposición $MSE(x)=\text{Varianza}+\text{Sesgo}^2+\sigma^2$, el término $\sigma^2$:",
   [r"Es el error irreducible: no depende del modelo", r"Disminuye al aumentar la complejidad del modelo",
    r"Es cero cuando se usa MCO", r"Mide cuánto varía $\hat f$ de muestra a muestra"], 0,
   r"$\sigma^2$ es la varianza del ruido $\epsilon$ en $y=f(x)+\epsilon$. Ningún modelo puede reducirlo; solo la varianza y el sesgo dependen de $\hat f$.")
mc("i3", "intro", 1,
   r"¿Cuál es el objeto de interés en un problema de <b>estimación</b>, frente a uno de predicción?",
   [r"$\hat\beta$, en lugar de $\hat Y$", r"$\hat Y$, en lugar de $\hat\beta$", r"El $R^2$ fuera de muestra", r"El error irreducible $\sigma^2$"], 0,
   r"En predicción nos interesa $\hat Y$ frente a $Y$ (dominio del ML). En estimación nos interesa $\hat\beta$ (dominio de la econometría).")
mc("i4", "intro", 2,
   r"En $Y=\beta_X X+\beta_W W+\epsilon$, ni $\hat\beta_X$ ni $\hat\beta_W$ son significativos. ¿Qué se puede concluir sobre la capacidad de $X$ y $W$ para predecir $Y$?",
   [r"Nada definitivo: pueden predecir bien $Y$ aunque cada coeficiente sea impreciso (p. ej., si $X$ y $W$ están muy correlacionadas)",
    r"Que $X$ y $W$ no sirven para predecir $Y$",
    r"Que el $R^2$ del modelo es cercano a cero",
    r"Que el modelo tiene sesgo por variable omitida"], 0,
   r"La significancia individual habla de la precisión de cada $\hat\beta$, no del poder predictivo conjunto. Con multicolinealidad los errores estándar se inflan, aunque $\hat Y$ sea muy preciso.")
num("i5", "intro", 2,
    r"Para un punto $x$: $E_D\hat f(x)=2.3$, $f(x)=2.0$, $E_D[(\hat f(x)-E_D\hat f(x))^2]=0.05$ y $\sigma^2=0.10$. Calcule $MSE(x)$.",
    0.24, 0.005,
    r"$MSE=\text{Var}+\text{Sesgo}^2+\sigma^2=0.05+(2.3-2.0)^2+0.10=0.05+0.09+0.10=0.24$.")
mc("i6", "intro", 2,
   r"El juez decide (a) si detiene al acusado antes del juicio y (b) si le pone brazalete electrónico. ¿Cuál decisión requiere conocer un <b>efecto causal</b>?",
   [r"(b): el brazalete puede cambiar la probabilidad de escape, $\partial Y/\partial X_0\neq0$",
    r"(a): la detención depende del efecto de la cárcel sobre la reincidencia",
    r"Ambas son problemas de predicción pura",
    r"Ninguna: ambas dependen solo del riesgo predicho"], 0,
   r"En (a) el pago depende del riesgo de escape si se libera: es predicción. En (b) el brazalete es una intervención que puede alterar $Y$; para decidir hay que saber cuánto la altera.")
mc("i7", "intro", 3,
   r"En $\hat f_{ML}=\arg\min_f \frac1n\sum_i (y_i-f(x_i))^2+\lambda R(f)$, ¿qué ocurre cuando $\lambda\to\infty$?",
   [r"Se elige la función más simple posible: varianza baja y sesgo alto",
    r"Se recupera MCO: sesgo cero y varianza alta",
    r"El MSE fuera de muestra se minimiza siempre",
    r"El error irreducible tiende a cero"], 0,
   r"Con $\lambda$ muy grande domina la penalidad: $R(f)$ se lleva a su mínimo (p. ej., todos los coeficientes en cero). El modelo casi no varía entre muestras, pero queda muy sesgado. Con $\lambda=0$ se vuelve a MCO.")
num("i8", "intro", 3,
    r"La función de pago es $\pi(X_0,Y)=-0.1X_0+10Y$ y el efecto causal de la acción es $\partial Y/\partial X_0=0.02$. Calcule $d\pi/dX_0$.",
    0.1, 0.002,
    r"$\frac{d\pi}{dX_0}=\frac{\partial\pi}{\partial X_0}+\frac{\partial\pi}{\partial Y}\frac{\partial Y}{\partial X_0}=-0.1+10(0.02)=0.1$. Vale la pena aumentar $X_0$ solo porque su efecto causal compensa el costo directo.")
mc("i9", "intro", 3,
   r"¿Por qué un algoritmo de ML puede superar a MCO <b>fuera de muestra</b>, aunque MCO tenga sesgo cero?",
   [r"Porque acepta algo de sesgo a cambio de una reducción mayor en la varianza",
    r"Porque elimina el error irreducible",
    r"Porque minimiza el MSE dentro de muestra mejor que MCO",
    r"Porque MCO es sesgado cuando $n$ es grande"], 0,
   r"MCO minimiza el error dentro de muestra y puede tener varianza alta. La regularización introduce sesgo, pero si la varianza cae más, el MSE fuera de muestra baja.")
mc("i10", "intro", 2,
   r"En $Y=\alpha D+g(W)+\epsilon$, ¿cuándo conviene usar ML en lugar de econometría clásica para recuperar $\alpha$?",
   [r"Cuando $W$ es grande y $g(W)$ es no lineal",
    r"Cuando $W$ es pequeño y el modelo es lineal",
    r"Cuando $D$ es asignado aleatoriamente y no hay $W$",
    r"Nunca: el ML solo sirve para predicción"], 0,
   r"Con pocos confusores y un modelo lineal, MCO funciona. Cuando hay muchos confusores o $g(W)$ es no lineal, el ML aproxima mejor $g(W)$, que es un problema de predicción.")

# ───────────────────────── MCO ─────────────────────────
mc("m1", "mco", 1,
   r"El estimador MCO es el análogo muestral de:",
   [r"El mejor predictor lineal (BLP), que minimiza $E[(Y-\beta'X)^2]$",
    r"La esperanza condicional $E[Y\mid W]$ para cualquier forma funcional",
    r"El estimador que minimiza el MSE fuera de muestra",
    r"El estimador Lasso con $\lambda>0$"], 0,
   r"El BLP resuelve $\min_\beta E[(Y-\beta'X)^2]$; MCO reemplaza la esperanza por el promedio muestral.")
mc("m2", "mco", 1,
   r"¿Qué describe el <b>overfitting</b>?",
   [r"Muy buen ajuste dentro de muestra y pésimo fuera de muestra",
    r"Mal ajuste tanto dentro como fuera de muestra",
    r"Un modelo con sesgo alto y varianza baja",
    r"Que el $R^2$ ajustado sea mayor que el $R^2$"], 0,
   r"Cuando $p/n$ no es pequeño, el modelo sigue el ruido de la muestra de entrenamiento. Las medidas de ajuste dentro de muestra se vuelven engañosas.")
mc("m3", "mco", 2,
   r"Notebook de predicción de salarios (GEIH Barranquilla). ¿Por qué el $R^2_{test}$ del modelo flexible es <b>negativo</b>?",
   [r"Su MSE fuera de muestra supera la varianza de $y$ alrededor de la media de entrenamiento: predice peor que la media. Es overfitting",
    r"Porque el modelo tiene menos regresores que el básico",
    r"Por un error de cálculo: el $R^2$ no puede ser negativo",
    r"Porque el modelo flexible tiene sesgo por variable omitida"], 0,
   r"$R^2_{test}=1-MSE_{test}/\overline{(y-\bar y_{train})^2}$. Con $MSE_{test}=1.871$ el cociente supera 1. Con 405 regresores el modelo sobreajustó: dentro de muestra $R^2=0.57$, pero fuera de muestra se desploma.",
   code="""              p  R2_sample  MSE_sample  R2_test  MSE_test
basic reg    94      0.51        0.18     0.443     0.192
flexible reg 405     0.57        0.16    -4.427     1.871""")
num("m4", "mco", 2,
    r"Con $n=500$ observaciones de entrenamiento, $p=100$ regresores y $MSE_{sample}=0.16$, calcule el MSE ajustado $\frac{n}{n-p}MSE_{sample}$.",
    0.2, 0.002,
    r"$\frac{500}{400}\times0.16=0.20$. El ajuste penaliza el número de parámetros porque el MSE dentro de muestra subestima el error de predicción.")
num("m5", "mco", 2,
    r"En la muestra de prueba, $MSE_{test}=0.1921$ y $\frac1{n_{test}}\sum(y_i-\bar y_{train})^2=0.345$. Calcule el $R^2_{test}$ (3 decimales).",
    0.443, 0.003,
    r"$R^2_{test}=1-0.1921/0.345=0.443$. Es el valor del modelo básico en el notebook de salarios.")
mc("m6", "mco", 3,
   r"Si $p>n$ en una regresión por MCO:",
   [r"$X'X$ es singular, $\beta$ no está identificado y se puede lograr un ajuste perfecto dentro de muestra",
    r"MCO sigue siendo consistente si los errores son homocedásticos",
    r"El $R^2$ dentro de muestra es cercano a cero",
    r"Basta usar errores estándar robustos"], 0,
   r"Con más parámetros que observaciones, $\sum_i X_iX_i'$ no es invertible y hay infinitas soluciones. El ajuste dentro de muestra puede ser perfecto y no dice nada sobre el desempeño fuera de muestra.")
mc("m7", "mco", 3,
   r"La varianza asintótica de MCO es $E[XX']^{-1}E[\epsilon^2XX']E[XX']^{-1}$. ¿A qué se reduce bajo <b>homocedasticidad</b>, $E[\epsilon^2\mid X]=\sigma^2$?",
   [r"$\sigma^2E[XX']^{-1}$", r"$\sigma^2E[XX']$", r"$E[\epsilon^2XX']^{-1}$", r"$\sigma^2/n$"], 0,
   r"Por expectativas iteradas, $E[\epsilon^2XX']=E[E[\epsilon^2\mid X]XX']=\sigma^2E[XX']$. Al reemplazar en la forma sándwich queda $\sigma^2E[XX']^{-1}$. Sin homocedasticidad se usa la versión robusta (HC1 en los notebooks).")
mc("m8", "mco", 1,
   r"¿Para qué usamos regresores transformados $X=T(W)$, como $W_1^2$ o interacciones?",
   [r"Para aproximar mejor $g(W)=E[Y\mid W]$ cuando la relación no es lineal",
    r"Para reducir el número de parámetros",
    r"Para eliminar la heterocedasticidad",
    r"Para garantizar que $E[\epsilon\mid X]=0$"], 0,
   r"El mejor predictor es la CEF $g(W)$. Con $\beta'T(W)$ la aproximamos de forma lineal en parámetros pero flexible en $W$, como en el ejemplo con $g(W)=\exp(4W)$ y polinomios.")
mc("m9", "mco", 3,
   r"Con $n$ fijo, se agregan regresores (polinomios de mayor grado) a una regresión por MCO. ¿Qué afirmación es correcta?",
   [r"El $R^2$ dentro de muestra nunca disminuye, pero el MSE fuera de muestra puede aumentar",
    r"Tanto el $R^2$ dentro como el de fuera de muestra aumentan siempre",
    r"El $R^2$ dentro de muestra disminuye cuando $p/n$ es grande",
    r"El MSE fuera de muestra es igual al de dentro de muestra si $p<n$"], 0,
   r"MCO minimiza la suma de residuales al cuadrado; añadir regresores nunca empeora el ajuste dentro de muestra. Pero la varianza crece y, cuando $p/n$ no es pequeño, $MSE_{sample}$ deja de aproximar $MSE_{pop}$.")

# ───────────────────────── LASSO ─────────────────────────
mc("l1", "lasso", 1,
   r"¿Qué distingue la penalidad de Lasso, $\lambda\sum_j|\beta_j|$?",
   [r"Encoge los coeficientes y lleva algunos <b>exactamente a cero</b>: selecciona variables",
    r"Encoge los coeficientes pero ninguno llega a cero",
    r"No introduce sesgo en los coeficientes",
    r"Garantiza seleccionar las variables con $\beta_j\neq0$ en la población"], 0,
   r"La penalidad en valor absoluto produce soluciones en las esquinas: algunos $\hat\beta_j=0$. Hay sesgo de encogimiento (shrinkage), y las variables seleccionadas no necesariamente son las correctas.")
mc("l2", "lasso", 1,
   r"¿Qué estimador se obtiene con Lasso si $\lambda=0$?",
   [r"MCO", r"Un modelo con solo intercepto", r"Post-Lasso", r"Ridge"], 0,
   r"Sin penalidad, el problema es minimizar la suma de residuales al cuadrado, que es MCO.")
num("l3", "lasso", 2,
    r"$\hat\theta$ es insesgado con $Var(\hat\theta)=v=1$, y $\theta=1$. Para $\tilde\theta=a\hat\theta$ con $a=0.5$, calcule $MSE(\tilde\theta)$.",
    0.5, 0.005,
    r"$MSE(\tilde\theta)=a^2v+(a-1)^2\theta^2=0.25(1)+0.25(1)=0.5$, menor que $MSE(\hat\theta)=v=1$. Un estimador sesgado puede tener menor MSE.")
num("l4", "lasso", 2,
    r"Con $\tilde\theta=a\hat\theta$, $a=0.5$ y $v=2$, $MSE(\tilde\theta)<MSE(\hat\theta)$ si y solo si $\theta^2<c$. Calcule $c$.",
    6.0, 0.02,
    r"La condición es $\theta^2<\frac{1+a}{1-a}v=\frac{1.5}{0.5}(2)=6$. El encogimiento conviene cuando el parámetro verdadero es pequeño en relación con la varianza.")
mc("l5", "lasso", 2,
   r"Salida de <code>cv.glmnet</code> en el notebook de salarios. ¿Qué representa la fila <code>1se</code> y por qué tiene menos variables?",
   [r"Es el $\lambda$ más grande cuyo MSE de validación cruzada está a menos de un error estándar del mínimo: más penalidad, modelo más parsimonioso",
    r"Es el $\lambda$ que minimiza el MSE de validación cruzada",
    r"Es el $\lambda$ teórico (plug-in)",
    r"Es el modelo estimado sin validación cruzada"], 0,
   r"<code>lambda.min</code>=0.0061 minimiza el CV-MSE (96 variables). <code>lambda.1se</code>=0.0297 es mayor: su MSE (0.2088) no es estadísticamente distinto del mínimo, y conserva solo 22 variables.",
   code="""      Lambda Index Measure      SE Nonzero
min 0.006116    40  0.1989 0.01071      96
1se 0.029739    23  0.2088 0.01237      22""")
mc("l6", "lasso", 2,
   r"Resultados fuera de muestra del notebook de salarios. ¿Qué se concluye?",
   [r"Lasso controla el sobreajuste del modelo flexible: su desempeño fuera de muestra es similar al de dentro de muestra",
    r"El modelo flexible por MCO es el mejor predictor",
    r"Lasso empeora la predicción frente al modelo básico",
    r"Las tres especificaciones son equivalentes"], 0,
   r"Con las mismas 405+ variables del modelo flexible, Lasso logra $MSE_{test}=0.187$, frente a 1.871 de MCO, y algo mejor que el modelo básico.",
   code="""                 MSE_test  R2_test
basic reg           0.192    0.443
flexible reg        1.871   -4.427
lasso regression    0.187    0.457""")
num("l7", "lasso", 3,
    r"Calcule el $\lambda$ teórico $\lambda=2c\hat\sigma\sqrt n\,z_{1-a/(2p)}$ con $c=1.1$, $a=0.05$, $\hat\sigma=0.4$, $n=400$ y $p=50$. Use $z_{0.9995}=3.2905$.",
    57.91, 0.3,
    r"$\lambda=2(1.1)(0.4)(20)(3.2905)=17.6\times3.2905\approx57.91$. Es la regla que usa <code>rlasso</code> del paquete <code>hdm</code>.")
mc("l8", "lasso", 3,
   r"¿Cuál es el propósito de MCO Post-Lasso?",
   [r"Reestimar por MCO con los regresores que Lasso seleccionó, para deshacer el sesgo de encogimiento",
    r"Elegir $\lambda$ por validación cruzada",
    r"Seleccionar variables que Lasso descartó",
    r"Corregir la heterocedasticidad de los errores"], 0,
   r"Post-Lasso usa la selección de Lasso ($b_j=0$ si $\hat\beta_j=0$) y estima sin penalidad los coeficientes restantes, lo que elimina el encogimiento hacia cero.")
mc("l9", "lasso", 3,
   r"$X_j$ y $X_k$ están muy correlacionadas y ambas tienen $\beta\neq0$ en la población. Lasso selecciona $X_j$ y lleva $\hat\beta_k$ a cero. ¿Qué implica?",
   [r"Es esperable: Lasso busca buena predicción y puede elegir una de las dos; la selección no identifica el modelo verdadero",
    r"Que $\beta_k=0$ en la población",
    r"Que $\lambda$ está mal elegido",
    r"Que $X_k$ no está correlacionada con $Y$"], 0,
   r"$\hat\beta_k=0$ cuando el beneficio marginal en predicción de $X_k$ es menor que $\lambda$. Si $X_j$ ya captura esa información, $X_k$ aporta poco. Por eso no se debe interpretar la selección de Lasso como causal ni como el modelo verdadero.")
num("l10", "lasso", 1,
    r"Validación cruzada con $K=5$: los MSE en cada fold son 0.20, 0.18, 0.22, 0.19 y 0.21. Calcule $CV_K$.",
    0.20, 0.002,
    r"$CV_K=\frac1K\sum_j MSE_j=(0.20+0.18+0.22+0.19+0.21)/5=0.20$.")

# ───────────────────────── CAUSAL ─────────────────────────
mc("c1", "causal", 1,
   r"¿Por qué no podemos calcular directamente $\Delta_i=Y_i(1)-Y_i(0)$?",
   [r"Porque para cada individuo solo observamos uno de los dos resultados potenciales",
    r"Porque $Y_i(1)$ y $Y_i(0)$ son siempre iguales",
    r"Porque el tratamiento nunca es aleatorio",
    r"Porque $D_i$ no es observable"], 0,
   r"Observamos $Y_i=Y_i(D_i)$: el resultado bajo el tratamiento recibido. El contrafactual no se observa, por eso nos enfocamos en parámetros poblacionales como el ATE.")
num("c2", "causal", 1,
    r"El 40% de la población recibe tratamiento ($\pi=0.4$), $ATT=2$ y $ATU=1$. Calcule el ATE.",
    1.4, 0.005,
    r"$ATE=\pi\,ATT+(1-\pi)ATU=0.4(2)+0.6(1)=1.4$.")
num("c3", "causal", 2,
    r"Con $ATE=1.4$, $\pi=0.4$, $ATT=2$, $ATU=1$ y sesgo de selección $E[Y(0)\mid D=1]-E[Y(0)\mid D=0]=0.5$, ¿cuánto vale la diferencia simple de medias (SDO)?",
    2.5, 0.005,
    r"$SDO=ATE+\text{sesgo de selección}+(1-\pi)(ATT-ATU)=1.4+0.5+0.6(1)=2.5$. La SDO sobrestima el ATE en 1.1.")
mc("c4", "causal", 2,
   r"Bajo asignación aleatoria, $(Y(1),Y(0))\perp D$. ¿Por qué entonces $SDO=ATE$?",
   [r"El sesgo de selección es cero y $ATT=ATU$, así que ambos términos de sesgo desaparecen",
    r"Porque $Y(1)=Y(0)$ para todos",
    r"Porque el tratamiento no tiene efecto",
    r"Porque se controla por todas las covariables"], 0,
   r"Con independencia, $E[Y(0)\mid D=1]=E[Y(0)\mid D=0]=E[Y(0)]$ y el efecto promedio es el mismo en tratados y no tratados.")
mc("c5", "causal", 2,
   r"Notebook de informalidad: se estima <code>lm(lw ~ inf)</code>. ¿Se puede interpretar el coeficiente como el efecto causal de la informalidad sobre el salario?",
   [r"No: los informales difieren en características que afectan el salario (educación, experiencia). Hay sesgo de selección, probablemente $E[Y(0)\mid D=1]<E[Y(0)\mid D=0]$",
    r"Sí, porque se usaron errores estándar robustos",
    r"Sí, porque la muestra es grande",
    r"No, porque el coeficiente es negativo"], 0,
   r"La SDO mezcla el ATE con el sesgo de selección. Los trabajadores informales probablemente tendrían menores salarios aun siendo formales, así que la SDO exagera la penalidad por informalidad. Los errores estándar no tienen nada que ver con la identificación.",
   code="""# salida ilustrativa
Coeficiente y error estándar robusto de Inf -0.6124 0.0187""")
num("c6", "causal", 3,
    r"En un RCT se estima $E[Y\mid D,X]=(\alpha_1+\alpha_2X)D+\beta X$ <b>sin centrar</b> $X$. Si $\hat\alpha_1=0.3$, $\hat\alpha_2=0.05$ y $\bar X=10$, ¿cuál es el ATE estimado?",
    0.8, 0.005,
    r"$ATE=E[\delta(X)]=\alpha_1+\alpha_2E[X]=0.3+0.05(10)=0.8$. Solo si $X$ está centrada ($E[X]=0$) el ATE es $\alpha_1$.")
mc("c7", "causal", 3,
   r"¿Qué verifica el <b>covariate balance check</b> en un RCT?",
   [r"Que $E[X\mid D=1]=E[X\mid D=0]$, como implica la asignación aleatoria",
    r"Que el efecto del tratamiento es homogéneo",
    r"Que las covariables predicen bien $Y$",
    r"Que no hay heterocedasticidad"], 0,
   r"Si $(Y(1),Y(0),X)\perp D$, las covariables pretratamiento deben tener la misma media en ambos grupos. Diferencias sistemáticas sugieren problemas en la aleatorización o desbalance por azar.")
mc("c8", "causal", 1,
   r"El ATT se define como:",
   [r"$E[Y_i(1)-Y_i(0)\mid D_i=1]$", r"$E[Y_i(1)-Y_i(0)]$", r"$E[Y_i(1)-Y_i(0)\mid D_i=0]$", r"$E[Y\mid D=1]-E[Y\mid D=0]$"], 0,
   r"El ATT es el efecto promedio en los tratados. La segunda opción es el ATE, la tercera el ATU y la cuarta la SDO.")
mc("c9", "causal", 3,
   r"En un RCT se agregan covariables <b>pretratamiento</b> a la regresión de $Y$ sobre $D$. ¿Qué ocurre con la estimación del ATE?",
   [r"Sigue siendo consistente y, si las covariables predicen $Y$, suele ser más precisa",
    r"Se vuelve sesgada porque las covariables son endógenas",
    r"Es necesario para identificar el ATE",
    r"Cambia el parámetro estimado de ATE a ATT"], 0,
   r"La aleatorización ya identifica el ATE. Las covariables pretratamiento reducen la varianza residual y mejoran la precisión; con interacciones permiten explorar heterogeneidad (CATE). Las covariables postratamiento sí podrían sesgar.")

# ───────────────────────── DOBLE LASSO ─────────────────────────
mc("d1", "dlasso", 1,
   r"Una buena variable de control en $Y=\alpha D+\beta'W+\epsilon$ cumple dos condiciones:",
   [r"Predice $Y$ (relevancia) y está correlacionada con $D$ (confunde)",
    r"No está correlacionada ni con $Y$ ni con $D$",
    r"Está correlacionada con $D$ pero no con $Y$",
    r"Es una variable postratamiento"], 0,
   r"Omitir una variable que cumple ambas condiciones produce sesgo por variable omitida e invalida la inferencia causal sobre $\alpha$.")
mc("d2", "dlasso", 1,
   r"En el notebook de brecha salarial, la regresión directa y el partialling-out dan el mismo coeficiente. ¿Qué resultado lo garantiza?",
   [r"El teorema de Frisch-Waugh-Lovell",
    r"La ortogonalidad de Neyman",
    r"El teorema del límite central",
    r"La asignación aleatoria de $M$"], 0,
   r"FWL: el coeficiente de $D$ en la regresión larga es igual al de regresar el residual de $Y$ (sobre $W$) en el residual de $D$ (sobre $W$).",
   code="""MCO directo:     Coef. M: -0.0883  | EE: 0.0161
Partialling-out: Coef. M: -0.0883  | EE: 0.0159""")
mc("d3", "dlasso", 2,
   r"¿Cuál es el problema de la selección simple (Lasso de $Y$ sobre $D$ y $W$, y luego MCO con las $W$ seleccionadas)?",
   [r"Descarta controles con efecto pequeño sobre $Y$ pero muy correlacionados con $D$, que generan sesgo por variable omitida",
    r"Selecciona demasiadas variables",
    r"Es computacionalmente costosa",
    r"Requiere que $p<n$"], 0,
   r"Lasso en la ecuación de $Y$ solo mira el poder predictivo para $Y$. Un confusor débil en $Y$ pero fuerte en $D$ queda fuera. Doble Lasso agrega la ecuación de $D$ para capturarlo.")
num("d4", "dlasso", 2,
    r"Doble Lasso con la especificación flexible da $\hat\alpha=-0.0918$ con EE $=0.0156$. Calcule el estadístico $t$ (2 decimales).",
    -5.88, 0.02,
    r"$t=-0.0918/0.0156=-5.88$. Se rechaza $H_0:\alpha=0$. Las mujeres ganan cerca de 9% menos por hora, condicional en los controles.",
    code="""Coef. M: -0.0918  | EE: 0.0156""")
mc("d5", "dlasso", 2,
   r"En el notebook, MCO con 471 regresores da $-0.088$ y doble Lasso $-0.0918$. ¿Por qué los resultados son tan parecidos?",
   [r"Porque $p/n$ es pequeño: con muchas observaciones, MCO todavía funciona bien",
    r"Porque Lasso no seleccionó ninguna variable",
    r"Porque doble Lasso siempre coincide con MCO",
    r"Porque la brecha salarial es cero"], 0,
   r"Doble Lasso es valioso cuando $p$ es grande respecto a $n$. Si $p/n$ es pequeño, MCO es preciso y ambos estimadores coinciden aproximadamente.",
   code="""MCO flexible:        Coef. M: -0.088   | EE: 0.0169
Doble Lasso flexible: Coef. M: -0.0918  | EE: 0.0156""")
mc("d6", "dlasso", 3,
   r"La ecuación de momentos $M(a,\eta)=E[(\tilde Y(\eta)-a\tilde D(\eta))\tilde D(\eta)]$ es Neyman ortogonal: $\partial_\eta M(\alpha,\eta_o)=0$. ¿Qué implica?",
   [r"El sesgo de regularización en $\hat\eta$ no se transmite a $\hat\alpha$, al menos a primer orden",
    r"$\hat\eta$ se estima sin sesgo",
    r"$\alpha$ no depende de $W$",
    r"No es necesario estimar $\eta$"], 0,
   r"$\alpha$ es localmente insensible a errores en los parámetros de molestia. En cambio, la selección simple usa $E[(Y-aD-b'W)D]$, con derivada $E[DW]\neq0$: no es ortogonal.")
mc("d7", "dlasso", 3,
   r"En el notebook de doble Lasso, sin sample splitting, se usa <code>rlasso</code> (λ teórico) en lugar de validación cruzada. ¿Por qué?",
   [r"La regla plug-in evita el sobreajuste en la primera etapa; con CV en muestras moderadas el error en $\check Y$ y $\check D$ puede afectar $\hat\alpha$ a primer orden",
    r"Porque CV no puede usarse con variables binarias",
    r"Porque el λ teórico siempre da menor MSE de predicción",
    r"Porque rlasso no encoge los coeficientes"], 0,
   r"La teoría de doble Lasso requiere buena predicción de $D$ y $Y$ sin sobreajuste, que la regla plug-in garantiza. Con CV la solución es hacer cross-fitting, que es DML.")
mc("d8", "dlasso", 3,
   r"Se incluye $W$ solo si su estadístico $t$ es grande y luego se reporta $\tilde\alpha$ como si el modelo fuera el verdadero. Si $\beta$ es pequeño pero distinto de cero y $D$ y $W$ están muy correlacionadas:",
   [r"La distribución de $\tilde\alpha$ no es normal (puede ser bimodal) y las pruebas e intervalos quedan distorsionados",
    r"$\tilde\alpha$ es consistente y normal",
    r"Solo aumenta la varianza, sin sesgo",
    r"El problema desaparece con errores robustos"], 0,
   r"La selección no es uniformemente consistente: con alta probabilidad omite $W$ y genera sesgo, y otras veces la incluye. La mezcla produce una distribución no normal. Es el problema de la inferencia post-selección.")
num("d9", "dlasso", 2,
    r"Con los residuales de la primera etapa, $\sum_i\check D_i\check Y_i=-3.6$ y $\sum_i\check D_i^2=40$. Calcule $\hat\alpha$.",
    -0.09, 0.001,
    r"$\hat\alpha=(\mathbb E_n[\check D^2])^{-1}\mathbb E_n[\check D\check Y]=-3.6/40=-0.09$. Los $1/n$ se cancelan.")
mc("d10", "dlasso", 1,
   r"¿Qué pasa si se agrega a la regresión una variable que <b>no predice</b> $Y$ (no cumple relevancia)?",
   [r"Deteriora la precisión de la estimación de $\alpha$",
    r"Mejora la precisión de la estimación de $\alpha$",
    r"Genera sesgo por variable omitida",
    r"No tiene ningún efecto"], 0,
   r"Una variable irrelevante no reduce la varianza residual y, si está correlacionada con $D$, aumenta la varianza de $\hat\alpha$. Una variable relevante, en cambio, mejora la precisión.")

# ───────────────────────── RANDOM FOREST ─────────────────────────
mc("r1", "rf", 1,
   r"¿Cómo predice un árbol de regresión en un punto $x$?",
   [r"Con el promedio de las $Y_i$ que caen en la misma hoja $L(x)$",
    r"Con una regresión lineal dentro de cada hoja",
    r"Con la mediana de todas las $Y_i$",
    r"Con el valor de $Y$ de la observación más cercana"], 0,
   r"$\hat\mu(x)=\sum_i\alpha_i(x)Y_i$ con $\alpha_i(x)=1\{X_i\in L(x)\}/|\{i:X_i\in L(x)\}|$: una función constante por tramos.")
mc("r2", "rf", 1,
   r"El bagging (promediar árboles construidos en submuestras) sirve principalmente para:",
   [r"Reducir la varianza", r"Reducir el sesgo", r"Seleccionar variables", r"Reducir el error irreducible"], 0,
   r"Un árbol profundo es muy inestable. Promediar $B$ árboles sobre submuestras suaviza la predicción y reduce la varianza.")
num("r3", "rf", 2,
    r"Un nodo de clasificación tiene una proporción de unos $\bar Y_A=0.3$. Calcule el índice de Gini $G(A)$.",
    0.42, 0.003,
    r"$G(A)=1-\bar Y_A^2-(1-\bar Y_A)^2=1-0.09-0.49=0.42$.")
num("r4", "rf", 3,
    r"Un nodo con $N_A=100$ y $\bar Y_A=0.5$ se divide en $A_1$ ($N=40$, $\bar Y=0.9$) y $A_2$ ($N=60$, $\bar Y=14/60$). Calcule la ganancia $I(A_1,A_2)=G(A)-qG(A_1)-(1-q)G(A_2)$ (3 decimales).",
    0.213, 0.003,
    r"$G(A)=0.5$, $G(A_1)=1-0.81-0.01=0.18$, $G(A_2)\approx0.358$, $q=0.4$. $I=0.5-0.4(0.18)-0.6(0.358)\approx0.213$.")
mc("r5", "rf", 2,
   r"Salida de <code>printcp</code> en el notebook. ¿Por qué se poda con el CP de la fila 8 y no con el último, que tiene menor <code>rel error</code>?",
   [r"Porque <code>rel error</code> es dentro de muestra y siempre baja; <code>xerror</code> es validado cruzado y es mínimo en la fila 8",
    r"Porque la fila 8 tiene el mayor número de cortes",
    r"Porque <code>xstd</code> es máximo en la fila 8",
    r"Porque el CP debe ser siempre mayor que 0.004"], 0,
   r"Más cortes siempre reducen el error dentro de muestra. El error de validación cruzada (<code>xerror</code>=0.56964) es mínimo con 7 cortes: <code>bestcp</code>=0.004767.",
   code="""          CP nsplit rel error  xerror     xstd
6  0.0081126      5   0.54656 0.57606 0.024852
7  0.0080498      6   0.53844 0.57351 0.024485
8  0.0047671      7   0.53039 0.56964 0.024347
9  0.0039436      8   0.52563 0.56969 0.024068
10 0.0038290     10   0.51774 0.56975 0.024091
11 0.0034930     11   0.51391 0.57014 0.024104""")
num("r6", "rf", 2,
    r"En la misma salida, <code>rel error</code> es el error relativo al nodo raíz. ¿Cuál es el $R^2$ dentro de muestra del árbol con 7 cortes (<code>rel error</code> = 0.53039)? Use 2 decimales.",
    0.47, 0.01,
    r"$R^2=1-\text{rel error}=1-0.53039\approx0.47$. Fuera de muestra el árbol podado logra $R^2=0.42$.",
    code="""Root node error: 1176.4/3122 = 0.37682
          CP nsplit rel error  xerror     xstd
8  0.0047671      7   0.53039 0.56964 0.024347""")
mc("r7", "rf", 3,
   r"En el notebook, <code>randomForest(..., mtry = p)</code> da $R^2=0.43$ y el bosque con <code>mtry</code> por defecto da $0.42$. ¿Qué explica la poca diferencia?",
   [r"Con <code>mtry = p</code> es bagging; con solo 4 predictores, sortear variables en cada nodo casi no descorrelaciona los árboles",
    r"Los dos modelos son idénticos",
    r"El bosque aleatorio no reduce varianza",
    r"<code>mtry</code> controla el número de árboles"], 0,
   r"El bosque aleatorio mejora al bagging al decorrelacionar los árboles (cada corte considera un subconjunto de variables). Con pocas variables esa ganancia es pequeña.")
mc("r8", "rf", 3,
   r"En regresión, maximizar $I(A_1,A_2)=E(A)-(qE(A_1)+(1-q)E(A_2))$ equivale a:",
   [r"Minimizar la varianza ponderada de los hijos, porque $E(A)$ no depende del corte",
    r"Maximizar la varianza de los hijos",
    r"Minimizar el índice de Gini",
    r"Maximizar el número de observaciones en $A_1$"], 0,
   r"$E(A)$ es la varianza del nodo padre, fija para todos los cortes. Solo el término de los hijos depende de $j$ y $s$.")
mc("r9", "rf", 1,
   r"Un árbol muy profundo (pocas observaciones por hoja), comparado con un bosque:",
   [r"Tiene sesgo bajo y varianza alta: sigue el ruido de los datos",
    r"Tiene sesgo alto y varianza baja",
    r"Tiene sesgo y varianza bajos",
    r"Siempre predice mejor fuera de muestra"], 0,
   r"Como en la figura de la presentación: un árbol profundo se ajusta al ruido; el promedio de $B$ árboles es mucho más estable.")
num("r10", "rf", 1,
    r"Una hoja contiene cuatro observaciones con $Y=2,3,4,7$. ¿Cuál es la predicción del árbol para un nuevo $x$ que cae en esa hoja?",
    4.0, 0.005,
    r"La predicción es el promedio de la hoja: $(2+3+4+7)/4=4$.")

# ───────────────────────── DML ─────────────────────────
mc("k1", "dml", 1,
   r"¿Cuál es el propósito de DML?",
   [r"Reducir el impacto de los parámetros fastidiosos (nuisance) estimados con ML sobre el parámetro de interés",
    r"Mejorar la predicción de $Y$",
    r"Seleccionar las variables del modelo verdadero",
    r"Reemplazar la asignación aleatoria"], 0,
   r"DML estima $\theta_0$ de $E[m(W,\theta_0,\eta_0)]=0$, donde $\eta_0$ (p. ej., $\ell(X)$ y $m(X)$) se estima con ML.")
mc("k2", "dml", 1,
   r"En cross-fitting, cada observación $i$ recibe su residual $\check Y_i$:",
   [r"De un modelo que no la usó para entrenarse",
    r"De un modelo entrenado con toda la muestra",
    r"De un modelo entrenado solo con su fold",
    r"Del promedio de $K$ modelos, incluido el que la usó"], 0,
   r"Para $i\in I_k$, $\check Y_i=Y_i-\hat\ell_{-k}(X_i)$, con $\hat\ell_{-k}$ estimado en $I_k^c$.")
mc("k3", "dml", 2,
   r"¿Cuál es la diferencia entre validación cruzada y cross-fitting?",
   [r"CV usa los folds para elegir hiperparámetros; cross-fitting los usa para construir residuales fuera de muestra",
    r"Son el mismo procedimiento con distinto nombre",
    r"Cross-fitting usa un solo fold de prueba",
    r"CV requiere ortogonalidad de Neyman"], 0,
   r"Pueden combinarse: dentro de $I_k^c$ se puede hacer CV para elegir $\lambda$ o la profundidad del bosque.")
mc("k4", "dml", 2,
   r"Sin cross-fitting, el término $\frac1{\sqrt n}\sum_i(\hat m(X_i)-m(X_i))\epsilon_i$ puede no desvanecerse. ¿Por qué?",
   [r"Si $\hat m$ se estima con las mismas observaciones, puede aprender el ruido y $\hat m(X_i)-m(X_i)$ queda correlacionado con $\epsilon_i$",
    r"Porque $\epsilon_i$ tiene media distinta de cero",
    r"Porque $m(X)$ es lineal",
    r"Porque $n$ es pequeño"], 0,
   r"Es el sesgo por sobreajuste. Con cross-fitting, $\hat m_{-k}$ es independiente de $(X_i,\epsilon_i)$ para $i\in I_k$, el término tiene media cero y desaparece si el ML es consistente.")
num("k5", "dml", 2,
    r"Con $n=1000$ y $K=5$ folds, ¿cuántas observaciones se usan para entrenar cada $\hat m_{-k}$?",
    800, 0.5,
    r"Cada $\hat m_{-k}$ se entrena en $I_k^c$, que tiene $n-n/K=1000-200=800$ observaciones.")
mc("k6", "dml", 3,
   r"¿Por qué se repite el cross-fitting $R$ veces y se reporta la mediana de los $\hat\beta_r$?",
   [r"Porque $\hat\beta$ depende del sorteo aleatorio de los folds",
    r"Para aumentar el tamaño de muestra",
    r"Porque la media de $\hat\beta_r$ siempre es sesgada",
    r"Para elegir el número de folds"], 0,
   r"Distintas particiones dan estimaciones algo distintas. Agregar sobre $R$ repeticiones reduce la dependencia de una partición particular.")
num("k7", "dml", 3,
    r"Tras cross-fitting: $\mathbb E_n[\check D\check Y]=0.12$, $\mathbb E_n[\check D^2]=0.4$, $\mathbb E_n[\check D^2\hat\epsilon^2]=0.05$ y $n=500$. Con $\hat V=(\mathbb E_n[\check D^2])^{-1}\mathbb E_n[\check D^2\hat\epsilon^2](\mathbb E_n[\check D^2])^{-1}$, calcule el error estándar $\sqrt{\hat V/n}$ (3 decimales).",
    0.025, 0.001,
    r"$\hat\beta=0.12/0.4=0.3$. $\hat V=0.05/0.16=0.3125$ y $se=\sqrt{0.3125/500}=0.025$.")
mc("k8", "dml", 3,
   r"Frente a un simple sample splitting (estimar nuisance en una mitad y $\beta$ en la otra), cross-fitting:",
   [r"Rota los papeles de los folds y usa las $n$ observaciones en la estimación final de $\beta$: es más eficiente",
    r"Elimina la necesidad de ortogonalidad de Neyman",
    r"Produce exactamente el mismo estimador",
    r"Requiere menos estimaciones de ML"], 0,
   r"Con sample splitting se pierde la mitad de la muestra en la etapa final. Cross-fitting obtiene residuales fuera de muestra para todas las observaciones.")
mc("k9", "dml", 2,
   r"Un estudiante hace partialling-out con un bosque aleatorio muy profundo, usando predicciones <b>dentro de muestra</b> para $\hat\ell(X)$ y $\hat m(X)$. ¿Qué problema tiene?",
   [r"Los residuales absorben poco del ruido porque el bosque lo aprendió: hay sesgo por sobreajuste en $\hat\beta$",
    r"Ninguno: el teorema de partialling-out lo garantiza",
    r"Solo aumenta el error estándar",
    r"El bosque no puede estimar $m(X)$ si $D$ es binaria"], 0,
   r"El partialling-out funciona con $\ell$ y $m$ verdaderos. Con predicciones sobreajustadas, los residuales $\check D$ son demasiado pequeños y correlacionados con el ruido. Cross-fitting lo corrige.")
mc("k10", "dml", 2,
   r"En el modelo parcial lineal $Y=\beta D+g(X)+\epsilon$, el partialling-out usa $\ell(X)=$",
   [r"$E[Y\mid X]$", r"$E[Y\mid D,X]$", r"$g(X)$", r"$E[D\mid X]$"], 0,
   r"$\ell(X)=E[Y\mid X]=\beta m(X)+g(X)$, no $g(X)$. Así $Y-\ell(X)=\beta(D-m(X))+\epsilon$.")
