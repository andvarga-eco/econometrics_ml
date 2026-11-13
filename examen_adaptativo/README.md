# Examen adaptativo · Econometría con ML

## Archivos

- `bank.py`: el banco de preguntas. Es el único archivo que necesitas editar.
- `template.html`: la página: diseño y motor adaptativo.
- `build.py`: une el banco y la plantilla, y genera `examen_adaptativo.html`.
- `examen_adaptativo.html`: la página generada. Se abre en cualquier navegador.

## Cómo editar preguntas

Cada pregunta es una llamada a `mc(...)` (opción múltiple) o `num(...)` (respuesta numérica) en `bank.py`:

```python
mc("l2", "lasso", 1,                      # id único, tema, nivel (1 básico, 2 intermedio, 3 avanzado)
   r"¿Qué estimador se obtiene con Lasso si $\lambda=0$?",   # enunciado
   [r"MCO", r"Un modelo con solo intercepto", r"Post-Lasso", r"Ridge"],  # opciones
   0,                                     # índice de la correcta (las opciones se barajan al mostrarse)
   r"Sin penalidad, el problema es ...",  # explicación
   code="...")                            # opcional: salida de R que se muestra en bloque de código

num("l10", "lasso", 1,
    r"Calcule $CV_K$ ...",
    0.20,      # respuesta
    0.002,     # tolerancia absoluta
    r"Explicación ...")
```

- Escribe las fórmulas en LaTeX entre `$...$` y usa cadenas `r"..."` para que las barras invertidas funcionen.
- Puedes usar HTML sencillo: `<b>`, `<code>`.
- Los temas válidos son `intro`, `mco`, `lasso`, `causal`, `dlasso`, `rf` y `dml`. Para agregar un tema, añádelo a `TOPICS`.
- Conviene tener al menos 3 preguntas por nivel en cada tema.

## Cómo regenerar la página

Desde esta carpeta:

```
python build.py
```

El script verifica que los id no se repitan y que temas, niveles y respuestas sean válidos, e imprime cuántas preguntas hay por tema y nivel. Después pídele a Claude que vuelva a publicar `examen_adaptativo.html` en el mismo enlace.

## Parámetros del simulacro (en `template.html`)

- `P1_PER_TOPIC = 2`: preguntas por tema en la primera ronda.
- `P2_EXTRA = 8`: preguntas dirigidas a los temas débiles.
- `MAX_PER_TOPIC = 5`: tope de preguntas por tema.
- `K = 1.2`: qué tan rápido se ajusta el nivel tras cada respuesta.
