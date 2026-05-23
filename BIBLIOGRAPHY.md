# Bibliografía Recomendada - MammoViz-TDA

Este documento centraliza las lecturas sugeridas en la hoja de ruta para dominar los tres pilares del proyecto: Análisis Topológico de Datos (TDA), Deep Learning en Imagen Médica y Procesamiento de Imágenes DICOM.

## 1. Análisis Topológico de Datos (TDA)
La base matemática para extraer características geométricas de las microcalcificaciones.

*   **A User’s Guide to Topological Data Analysis**
    *   *Autores:* Lum et al. (2013).
    *   *Enfoque:* Introducción práctica al TDA y cómo se aplica para entender datos complejos. Ideal para las primeras 15 páginas.
*   **An Introduction to Topological Data Analysis**
    *   *Autores:* Chazal & Michel (2021).
    *   *Enfoque:* Un enfoque más riguroso y moderno. Recomendado leer las primeras 20 páginas para entender filtraciones de Vietoris-Rips.
*   **Documentación de GUDHI 3.10**
    *   *Recurso:* [gudhi.inria.fr](https://gudhi.inria.fr/doc/latest/)
    *   *Conceptos clave:* `Rips_complex`, `Simplex_tree`, `persistent_cohomology`.

## 2. Deep Learning e Imagen Médica
Para entender cómo las CNNs clasifican patologías y cómo interpretar sus decisiones.

*   **Deep Learning (Libro)**
    *   *Autores:* Ian Goodfellow, Yoshua Bengio y Aaron Courville.
    *   *Lectura:* Capítulo 9 (Convolutional Networks).
*   **Deep Learning for Mammography: Additive Learning for Classification of Short-term Risk**
    *   *Autores:* Shen et al. (2019).
    *   *Enfoque:* Caso de estudio real sobre clasificación en mamografías.
*   **Papers with Code: CBIS-DDSM**
    *   *Recurso:* [paperswithcode.com/dataset/cbis-ddsm](https://paperswithcode.com/dataset/cbis-ddsm)
    *   *Objetivo:* Ver el estado del arte actual en el dataset que estamos usando.

## 3. IA Explicable (XAI)
Para entender por qué el modelo toma sus decisiones.

*   **Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization**
    *   *Autores:* Selvaraju et al. (2017).
    *   *Enfoque:* La técnica principal que implementaremos para resaltar zonas críticas en la imagen.
*   **Axiomatic Attribution for Deep Networks (Integrated Gradients)**
    *   *Autores:* Sundararajan et al. (2017).
    *   *Enfoque:* Una técnica más avanzada y teórica para atribución de importancia de píxeles.
*   **UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction**
    *   *Autores:* McInnes et al. (2018).
    *   *Lectura:* Secciones 1 y 3.

## 4. Estándares y Herramientas Técnicas
Lecturas esenciales para la implementación en C++.

*   **ITK Software Guide (Vol. 1)**
    *   *Lectura:* Capítulo 7 (Reading and Writing Images). Fundamental para el manejo de DICOM.
*   **Digital Mammography: A Practical Approach**
    *   *Enfoque:* Entender la parte clínica: qué es una calcificación benigna vs maligna desde el punto de vista de un radiólogo.
*   **DICOM Standard PS 3.3**
    *   *Lectura:* Image Module. Referencia técnica para entender los metadatos de las imágenes médicas.

---
*Nota: Este archivo se irá actualizando a medida que avancen las fases del proyecto.*
