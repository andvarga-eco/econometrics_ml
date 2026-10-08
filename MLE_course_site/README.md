# Econometría con Machine Learning — sitio del curso (propuesta en Quarto)

Materiales del curso de Econometría de la Maestría en Economía de la Universidad del Norte.

## Ver la propuesta sin instalar nada

Abre `_site/index.html` en el navegador. Es la versión ya renderizada (la carpeta `_site/` se regenera, no se edita a mano).

## Estructura

```
_quarto.yml              Configuración del sitio (título, menú, pie)
index.qmd                Portada del curso
sesiones.yml             ← LISTA DE SESIONES: aquí se agregan clases
referencias.bib          Bibliografía común (citas con @clave)
assets/
  sitio.scss             Estilo de la página web
  slides.scss            Tema de TODAS las presentaciones
  sesiones.ejs           Plantilla de las tarjetas de sesión
  logo-uninorte.png
presentaciones/
  _metadata.yml          Opciones comunes a todas las presentaciones
  introduccion/index.qmd
  inferencia-causal/index.qmd   (incluye una guía rápida de sintaxis)
notebooks/               .ipynb que se abren en Colab
.github/workflows/publish.yml   Publicación automática en GitHub Pages
```

## Flujo de trabajo

1. **Nueva clase:** crea `presentaciones/NOMBRE/index.qmd` con
   ```yaml
   ---
   title: "Título"
   subtitle: "Subtítulo"
   ---
   ```
   y escribe las diapositivas en Markdown (`## Título` = nueva diapositiva, `# Título` = separador de sección).
2. **Agregarla al sitio:** añade un bloque en `sesiones.yml` (número, título, temas, ruta, enlace de Colab).
3. **Ver mientras escribes:** `quarto preview` (se recarga sola al guardar).
4. **Publicar:** `git push`. La GitHub Action renderiza y publica.

## Atajos en las presentaciones

| Tecla | Acción |
|---|---|
| `M` | Menú con el índice de diapositivas |
| `F` | Pantalla completa |
| `S` | Vista del orador (notas) |
| `B` / `C` | Pizarra / dibujar sobre la diapositiva |
| `E` y luego `Ctrl+P` | Exportar a PDF |

## Instalación (una sola vez)

- Quarto: <https://quarto.org/docs/get-started/> (la propuesta se probó con la versión 1.7.32).
- En GitHub: *Settings → Pages → Source: GitHub Actions*.
