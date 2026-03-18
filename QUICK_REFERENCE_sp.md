# Guía de Referencia Rápida de SAIS

## ¿Qué es SAIS?

**SAIS** = Sistema Quirúrgico IA (Surgical AI System)

Un sistema de aprendizaje profundo que analiza videos quirúrgicos para identificar automáticamente:
- Pasos quirúrgicos (fases de la operación)
- Gestos quirúrgicos (movimientos manuales específicos)
- Habilidad quirúrgica (calidad de las acciones)
- Importancia del fotograma (qué fotogramas del video importan más)

**Tecnología**: Transformador de Visión (ViT) + Aprendizaje Contrastivo Supervisado

**Artículo Principal**: [Nature Biomedical Engineering - 2023](https://www.nature.com/articles/s41551-023-01010-8)

---

## Inicio Rápido (5 Comandos)

```bash
# 1. Clonar y configurar
git clone https://github.com/danikiyasseh/SAIS.git
cd SAIS
conda create -n SAIS python=3.9.7 -y && conda activate SAIS
pip install -r requirements.txt && pip install -e .

# 2. Descargar pesos de DINO (CRÍTICO)
mkdir -p SAIS/scripts/dino-main/outputs
# Descargar de: https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth
# O usar: wget [URL] -O SAIS/scripts/dino-main/outputs/dino_deitsmall16_pretrain.pth

# 3. Modificar transformador de PyTorch (CRÍTICO)
# Editar: {CONDA_ENV}/lib/python3.9/site-packages/torch/nn/modules/transformer.py
# Líneas 181 y 294: Retornar pesos de atención junto con la salida

# 4. Colocar video en carpeta videos/
mkdir -p SAIS/videos
# Copiar su_video.mp4 en SAIS/videos/

# 5. Ejecutar inferencia end-to-end
cd SAIS
bash ./main.sh -f su_video
```

**Salida**: Predicciones en la carpeta `predictions/` con etiquetas de gestos y puntuaciones de confianza

---

## Estructura de Archivos

```
SAIS/
├── videos/                  # ENTRADA: Sus videos quirúrgicos aquí
├── SAIS/
│   ├── main.sh             # Punto de entrada - ejecutar esto
│   ├── scripts/
│   │   ├── run_experiments.py        # Orquestador de inferencia
│   │   ├── extract_representations.py # Extracción de características
│   │   ├── generate_paths.py         # Mapeo de rutas
│   │   ├── train.py / perform_training.py
│   │   ├── prepare_model.py          # Constructor del modelo
│   │   ├── prepare_dataset.py        # Cargador de datos
│   │   ├── dino-main/               # Módulo extractor de características
│   │   │   └── outputs/
│   │   │       └── dino_deitsmall16_pretrain.pth  # REQUERIDO
│   │   └── video_to_frames.sh
│   └── params/
│       └── Fold_0/
│           ├── params.zip            # Pesos del modelo
│           └── prototypes.zip        # Prototipos de clases
├── images/                 # Generado: Fotogramas extraídos del video
├── flows/                  # Generado: Mapas de flujo óptico
├── paths/                  # Generado: CSV con rutas de fotogramas/flujos
├── results/                # Generado: Características extraídas (HDF5)
└── predictions/            # SALIDA: Predicciones finales
```

---

## 🔄 Pipeline de Flujo de Datos

```
Video Quirúrgico (MP4)
    ↓
[video_to_frames.sh] → Extraer 2000 fotogramas @ 20 FPS
    ↓
[generate_paths.py] → Crear CSV de rutas
    ↓
[RAFT] → Flujo óptico (movimiento)     [DINO] → Características RGB
    ↓                                      ↓
[Almacenamiento HDF5] ←────────────────┘
    ↓
[prepare_dataset.py] → Cargar y dividir en fragmentos de 5 fotogramas
    ↓
[Modelo ViT] → Procesar a través de capas de atención
    ↓
[Prototipos] → Clasificar cada fragmento
    ↓
[process_inference_results.py] → Agregar a etiquetas de acción
    ↓
PREDICCIONES: Tipo de gesto, confianza, importancia del fotograma
```

---

## Parámetros de Configuración Clave

| Parámetro | Qué hace | Valor Típico |
|-----------|----------|-------------|
| `-f {videoname}` | Cuál video procesar | `surgery_001` |
| `-m ViT` | Arquitectura del modelo | Siempre `ViT` |
| `-mod RGB-Flow` | Usar RGB + flujo óptico | Recomendado |
| `-nc {num_classes}` | Número de tipos de gestos | 2-10 típicamente |
| `-bs {batch_size}` | Tamaño del lote | 2 (inferencia) / 8+ (entrenamiento) |
| `-t Prototypes` | Método de aprendizaje | Siempre `Prototypes` |
| `--inference` | Ejecutar en modo inferencia | Añadir esta bandera para predicciones |
| `-e {epochs}` | Épocas para entrenamiento | 50 típicamente |
| `-f {folds}` | Validación cruzada K-fold | 5 típicamente |

---

## Relaciones entre Componentes

```
                    ORQUESTADOR
                   run_experiments.py
                    /   |   |   \
                   /    |   |    \
          [MODELO]  [DATOS] [ENTRENAMIENTO] [INFERENCIA]
            /           |         |        \
           /            |         |         \
      prepare_model   prepare_  perform_    cargar params/
         .py           dataset   training    prototipos
                        .py      .py
                        
      ↓ ENTRADA         ↓         ↓ PÉRDIDA   ↓ SALIDA
      
   Características  Fragmentos  Pérdida      Predicciones
   HDF5 + Rutas    Etiquetados Contrastiva   + Atención
```

---

## Flujos de Trabajo Comunes

### Flujo de Trabajo 1: Analizar Nuevo Video Quirúrgico (Solo Inferencia)

```bash
# Requisitos previos:
# ✓ Configuración del entorno (Pasos 1-2 del Inicio Rápido)
# ✓ Pesos de DINO descargados
# ✓ Modelo pre-entrenado en params/Fold_0/

# Ejecutar:
cp su_video_quirurgico.mp4 SAIS/videos/
bash SAIS/main.sh -f su_video_quirurgico

# Resultados:
# Verificar: SAIS/predictions/
# Contiene: etiquetas de gestos, puntuaciones de confianza, importancia del fotograma
```

### Flujo de Trabajo 2: Entrenar Modelo en Su Conjunto de Datos

```bash
# 1. Preparar conjunto de datos:
#    - Extraer videos en images/
#    - Crear CSV de anotaciones: annotations/dataset_annotations.csv
#    - Formato: video_name, frame_num, gesture_label

# 2. Ejecutar extracción de características (igual que inferencia)
bash SAIS/main.sh -f video_1
bash SAIS/main.sh -f video_2
# ... repetir para todos los videos

# 3. Entrenar modelo:
python -m torch.distributed.launch \
  SAIS/scripts/run_experiments.py \
  -p ./SAIS/ \
  -data Custom_Gestures \
  -d Custom \
  -m ViT \
  -enc ViT_SelfSupervised_ImageNet \
  -t Prototypes \
  -mod RGB-Flow \
  -dim 384 \
  -bs 8 \
  -lr 0.1 \
  -nc 10 \
  -bc \
  -sa \
  -e 50 \
  -f 5

# 4. Modelo guardado en: SAIS/params/Fold_X/
```

### Flujo de Trabajo 3: Evaluar Desempeño del Modelo

```bash
# Igual que el entrenamiento pero con más datos de validación
# Métricas calculadas: Exactitud, Puntuación F1, Matriz de confusión
# Resultados guardados en: ptlflow_logs/

cat ptlflow_logs/log_run.txt
```

---

## Stack Técnico

| Componente | Tecnología | Propósito |
|-----------|-----------|---------|
| **Procesamiento de Video** | ffmpeg, OpenCV | Extracción de fotogramas, I/O de video |
| **Extracción de Características** | DINO ViT-small | Representaciones visuales auto-supervisadas |
| **Flujo Óptico** | RAFT (ptlflow) | Estimación de movimiento |
| **Modelado Temporal** | PyTorch Transformer | Procesamiento de secuencias temporales |
| **Paradigma de Aprendizaje** | Aprendizaje Contrastivo Supervisado | Aprendizaje métrico con prototipos |
| **Aprendizaje Profundo** | PyTorch 1.8.0 | Entrenamiento e inferencia |
| **Entrenamiento Distribuido** | torch.distributed.launch | Soporte multi-GPU |

---

## Solución de Problemas

| Problema | Causa | Solución |
|---------|-------|-----|
| `ModuleNotFoundError: dino` | Ruta no configurada | Verificar sys.path en extract_representations.py |
| `CUDA out of memory` | Tamaño de lote muy grande | Reducir `--batch_size_per_gpu` |
| `RuntimeError: attention` | Transformador de PyTorch no modificado | Seguir Paso 3 en Inicio Rápido |
| `FileNotFoundError: dino_weights.pth` | Pesos no descargados | Descargar y colocar en carpeta correcta |
| `KeyError` en carga de datos | Formato de características incorrecto | Asegurar que los archivos HDF5 se creen exitosamente |
| No se generan predicciones | Falta params.zip/prototypes.zip | Descargar modelos pre-entrenados o entrenar |

---

## Información Clave

1. **Pre-entrenamiento es Crítico**: Las características de DINO están pre-entrenadas en ImageNet. Este aprendizaje transferido es lo que hace funcionar SAIS.

2. **Fusión Multimodal**: Las características RGB + Flujo capturan tanto apariencia como movimiento, crucial para la comprensión quirúrgica.

3. **Interpretabilidad**: Los mapas de atención del transformador muestran cuáles fotogramas impulsan cada predicción.

4. **Generalización**: Modelo probado en 3 hospitales, 2 continentes, muestra fuerte transferencia entre dominios.

5. **Aprendizaje Contrastivo Supervisado**: Aprende prototipos discriminativos por clase de gesto en lugar de solo regresión softmax.

---

## Interpretación de Salida

**Ejemplo de Salida de Predicción** (formato CSV):

```
video_name,action_id,start_frame,end_frame,gesture_label,confidence,frame_importance
surgery_001,1,0,49,knot_tying,0.92,"[0.1, 0.2, 0.85, 0.9, 0.95]"
surgery_001,2,50,99,tissue_grasping,0.88,"[0.3, 0.7, 0.6, 0.4, 0.2]"
surgery_001,3,100,149,suturing,0.95,"[0.05, 0.15, 0.92, 0.88, 0.91]"
```

- **gesture_label**: Gesto quirúrgico predicho (clases entrenadas)
- **confidence**: Puntuación de probabilidad (0-1)
- **frame_importance**: Pesos de atención por fotograma en acción (muestra cuáles fotogramas importan más)

---

## Recursos de Aprendizaje

- **[Artículo Original de SAIS](https://www.nature.com/articles/s41551-023-01010-8)** - Detalles técnicos completos
- **[Transformadores de Visión (ViT)](https://arxiv.org/abs/2010.11929)** - Arquitectura central
- **[DINO: ViTs Auto-supervisados](https://github.com/facebookresearch/dino)** - Extractor de características
- **[Flujo Óptico RAFT](https://github.com/princeton-vl/RAFT)** - Estimación de movimiento
- **[Aprendizaje Contrastivo Supervisado](https://arxiv.org/abs/2004.11362)** - Paradigma de aprendizaje

---

## Soporte y Problemas

- **GitHub**: [danikiyasseh/SAIS](https://github.com/danikiyasseh/SAIS)
- **Citación**: Ver README del repositorio para referencia BibTeX
- **Licencia**: CC BY-NC 4.0 (Uso no comercial)

---

## Consejos para el Éxito

1. **GPU de Buena Calidad es Esencial**: Necesita al menos 8 GB VRAM para extracción de características
2. **Preprocesar Videos**: Asegurar tasas de fotogramas consistentes (20 FPS típico en videos quirúrgicos)
3. **Calidad de Anotación**: Para entrenamiento, etiquetas precisas son críticas
4. **Validación Cruzada**: Usar validación cruzada 5-fold para estimaciones robustas
5. **Monitorear Entrenamiento**: Ver curvas de pérdida en ptlflow_logs para convergencia
6. **Congelar Codificador**: Generalmente mejor congelar pesos de DINO (encontrado durante desarrollo)
