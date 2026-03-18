# Análisis del Sistema SAIS: Arquitectura, Componentes y Guía de Ejecución

## 📋 Resumen Ejecutivo

**SAIS** (Sistema de Inteligencia de Actividad Quirúrgica) es un sistema de aprendizaje profundo que decodifica videos quirúrgicos para identificar automáticamente y analizar la actividad quirúrgica intraoperatoria. Aprovecha Transformadores de Visión (ViT) con aprendizaje contrastivo supervisado para analizar videos de cirugías robóticas y proporcionar información sobre:

- **Pasos Quirúrgicos**: Identificación de fases/pasos dentro de un procedimiento
- **Gestos Quirúrgicos**: Reconocimiento de movimientos manuales y acciones específicos
- **Habilidades Quirúrgicas**: Evaluación de la calidad y precisión de las acciones quirúrgicas
- **Importancia del Fotograma**: Destacar qué fotogramas de video son más importantes para cada predicción

El sistema se generaliza entre videos, cirujanos, hospitales y procedimientos quirúrgicos.

---

## 🏗️ Arquitectura del Sistema

### Flujo de Datos de Alto Nivel

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ENTRADA: Video Quirúrgico                        │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Etapa 1: PREPROCESAMIENTO DE VIDEO                                 │
├─────────────────────────────────────────────────────────────────────┤
│ • Extraer fotogramas del video (video_to_frames.sh)                │
│ • Generar mapas de flujo óptico (ptlflow RAFT)                     │
│ • Crear asignaciones de rutas para procesamiento eficiente         │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
        ┌──────────────────────┐  ┌──────────────────────┐
        │  Fotogramas RGB      │  │  Mapas de Flujo      │
        │  (images/)           │  │  Óptico (flows/)     │
        └─────────┬────────────┘  └──────────┬───────────┘
                  │                          │
                  │                          ▼
                  │      ┌─────────────────────────────────────────┐
                  │      │  Etapa 2: EXTRACCIÓN DE CARACTERÍSTICAS │
                  │      ├─────────────────────────────────────────┤
                  │      │ • Usar DINO ViT-small (auto-supervisado)│
                  │      │ • Extraer características espaciales     │
                  │      │ • Salida: Embeddings 384-dim (h5)      │
                  │      │ • Resultados en directorio results/    │
                  │      └─────────────────────────────────────────┘
                  │                          │
                  └────────────┬─────────────┘
                               ▼
        ┌──────────────────────────────────────────────┐
        │  Características RGB + Flujo                 │
        │  (Vectores 384-dim por fotograma)            │
        └───────────────────┬──────────────────────────┘
                            │
                            ▼
        ┌──────────────────────────────────────────────┐
        │  Etapa 3: MODELADO TEMPORAL Y CLASIFICACIÓN │
        ├──────────────────────────────────────────────┤
        │ • Segmentar características en fragmentos    │
        │ • Pasar a través del Codificador Transformer│
        │ • Aplicar Aprendizaje Basado en Prototipos │
        │ • Pérdida Contrastiva Supervisada           │
        │ • Salida: Predicciones por fragmento        │
        │ • Extraer: Mapas de atención                │
        └───────────────────┬──────────────────────────┘
                            │
                            ▼
        ┌──────────────────────────────────────────────┐
        │  PREDICCIONES E INTERPRETABILIDAD            │
        ├──────────────────────────────────────────────┤
        │ • Clasificaciones de gestos                  │
        │ • Predicciones de pasos quirúrgicos          │
        │ • Puntuaciones de evaluación de habilidades │
        │ • Puntuaciones de importancia del fotograma │
        └─────────────────────────────────────────────┘
```

---

## 🔧 Descripción de Componentes

### 1. **Componentes de Preprocesamiento de Video**

#### `video_to_frames.sh`
- **Propósito**: Convertir video MP4 en imágenes de fotogramas individuales
- **Salida**: Fotogramas almacenados en directorio `images/{videoname}/`
- **Dependencias**: ffmpeg vía imageio

#### `generate_paths.py`
- **Propósito**: Crear archivos CSV con rutas a fotogramas y archivos de flujo óptico
- **Salidas**: 
  - `paths/{videoname}_FramePaths.csv`
  - `paths/{videoname}_FlowPaths.csv`
- **Parámetros clave**: Nombre de video, ruta raíz

---

### 2. **Componentes de Extracción de Características**

#### `extract_representations.py` + Modelo DINO
- **Propósito**: Extraer características profundas de fotogramas usando un Transformador de Visión pre-entrenado
- **Extractor de Características**: DINO (ViT-small-16)
  - Pre-entrenamiento auto-supervisado en imágenes naturales
  - Proporciona comprensión espacial rica
  - Salida: Embeddings de 384 dimensiones por fotograma
- **Procesa**:
  - Características de fotogramas RGB
  - Características de flujo óptico
- **Ubicación de Salida**: Directorio `results/` (formato HDF5)
- **Parámetros clave**:
  ```
  --arch vit_small
  --patch_size 16
  --batch_size_per_gpu 1024 (RGB) / 256 (Flow)
  --save_type h5
  --optical_flow / --optical_flow_to_reps
  ```

#### Pesos Pre-entrenados de DINO
- **Archivo**: `dino_deitsmall16_pretrain.pth` (~350 MB)
- **Ubicación**: `scripts/dino-main/outputs/`
- **Fuente**: Repositorio DINO de Facebook Research
- **Requerido**: Debe descargarse manualmente antes de la extracción de características

---

### 3. **Generación de Flujo Óptico**

#### ptlflow con Modelo RAFT
- **Propósito**: Generar mapas de flujo óptico (movimiento) entre fotogramas
- **Modelo**: RAFT (Transformadas de Campo de Todos los Pares Recurrente)
- **Salida**: Vectores de flujo almacenados junto a características de fotogramas
- **Usado por**: Modelado temporal para capturar información de movimiento

---

### 4. **Componentes de Arquitectura del Modelo**

#### Clases Principales (en `prepare_model.py`)

**Codificador Transformador de Visión**
- Dimensión: 384 (de DINO)
- Pre-entrenado: Sí (congelado o ajustable)
- Fusión multimodal: Concatenación RGB + Flujo Óptico

**Modelado Temporal (Codificador Transformador)**
- Número de capas: Configurable
- Mecanismos de auto-atención
- Ponderación de importancia del fotograma vía mapas de atención

**Cabeza de Clasificación Basada en Prototipos**
- Usa aprendizaje contrastivo supervisado
- Aprende prototipos (ejemplares representativos) por clase
- Permite predicciones interpretables

---

### 5. **Pipeline de Entrenamiento e Inferencia**

#### `train.py` + `run_experiments.py`
**Orquestador principal** que:
1. Carga datos vía `prepare_dataset.py`
2. Crea modelo vía `prepare_model.py`
3. Ejecuta entrenamiento vía `perform_training.py`
4. Calcula métricas vía `prepare_miscellaneous.py`

**Características clave**:
- Entrenamiento distribuido multi-GPU (torch.distributed.launch)
- Validación cruzada basada en fold
- Soporte flexible de tareas: Clasificación, Reconocimiento de Gestos, Evaluación de Habilidades
- Modo inferencia: Cargar modelos entrenados y predecir en nuevos videos

**Parámetros**:
- Modelo: Modelo temporal basado en ViT
- Tarea: Prototipos (aprendizaje contrastivo)
- Modalidades: RGB-Flow (fusión multimodal)
- Tamaño de lote: 2 (inferencia), mayor para entrenamiento
- Velocidad de aprendizaje: 0.1 (adaptativo)
- Número de clases: Específico de tarea (2-10)

---

### 6. **Procesamiento de Resultados**

#### `process_inference_results.py`
- **Propósito**: Post-procesar predicciones del modelo en predicciones válidas
- **Convierte**: Predicciones a nivel de fotograma en predicciones a nivel de fragmento/acción
- **Salida**: Resultados finales listos para interpretación

---

## 📊 Diagrama de Flujo de Datos

```
Video Quirúrgico (.mp4)
        │
        ├─► video_to_frames.sh ──► images/{video_name}/*.jpg
        │
        └─► generate_paths.py ──► paths/{video_name}_FramePaths.csv
                                  └─► paths/{video_name}_FlowPaths.csv
                                      
                                      │
                    ┌─────────────────┼─────────────────┐
                    │                 │                 │
                    ▼                 ▼                 │
        ┌─────────────────────┐  ┌──────────────────┐  │
        │ Extraer RGB         │  │ Extraer Flujo    │  │
        │ Características     │  │ Óptico vía RAFT  │  │
        │ vía DINO            │  │ (extract_reps.py)│  │
        │ (extract_reps.py)   │  │                  │  │
        └─────────┬───────────┘  └────────┬─────────┘  │
                  │                       │            │
                  └───────────┬───────────┘            │
                              ▼                        │
                ┌────────────────────────────┐         │
                │ Resultados: RGB + Flujo    │         │
                │ Características (HDF5)     │         │
                │ results/                   │         │
                └────────────┬───────────────┘         │
                             │                        │
                             │◄───────────────────────┘
                             │
                             ▼
        ┌────────────────────────────────────┐
        │ prepare_dataset.py: Cargar caract. │
        │ Crear cargador con fragmentos      │
        └────────────┬───────────────────────┘
                     │
                     ▼
        ┌────────────────────────────────────┐
        │ prepare_model.py: Construir modelo │
        │ ViT Encoder + RNN + Prototipos    │
        └────────────┬───────────────────────┘
                     │
                     ▼
        ┌────────────────────────────────────┐
        │ run_experiments.py / train.py:      │
        │ Modo Entrenamiento o Inferencia    │
        │ Cargar params/prototipos de        │
        │ params/{Fold_X}/                   │
        └────────────┬───────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
    Modo Inferencia      Modo Entrenamiento
    (bandera --inference) (acumular métricas)
        │                         │
        └─────────┬───────────────┘
                  │
                  ▼
    ┌────────────────────────────────────┐
    │ process_inference_results.py        │
    │ Convertir predicciones de fotograma│
    │ en predicciones de acción          │
    └────────────┬───────────────────────┘
                 │
                 ▼
    ┌───────────────────────────────────────────┐
    │ SALIDA FINAL                              │
    │ • Clasificaciones de gestos por acción   │
    │ • Segmentación de pasos quirúrgicos      │
    │ • Puntuaciones de habilidad              │
    │ • Importancia del fotograma (atención)   │
    └───────────────────────────────────────────┘
```

---

## 🚀 Guía de Compilación y Ejecución

### Requisitos Previos

1. **Requisitos del Sistema**
   - GPU con soporte CUDA (se recomienda NVIDIA)
   - Al menos 16 GB VRAM para extracción de características
   - 8+ GB RAM para entrenamiento del modelo
   - ~100 GB espacio en disco por conjunto de datos de video quirúrgico

2. **Requisitos de Software**
   - Python 3.9.7
   - PyTorch 1.8.0 compatible con CUDA 11.x
   - ffmpeg (para procesamiento de video)

---

### Paso 1: Clonar y Configurar Entorno

```bash
# Clonar repositorio
git clone https://github.com/danikiyasseh/SAIS.git
cd SAIS

# Crear entorno conda
conda create -n SAIS python=3.9.7 -y
conda activate SAIS

# Instalar dependencias
pip install -r requirements.txt
pip install -e .
```

#### Desglose de Dependencias:
| Paquete | Propósito |
|---------|---------|
| torch==1.8.0 | Marco de aprendizaje profundo |
| torchvision==0.9.0 | Utilidades de visión por computadora |
| opencv-python | Procesamiento de imágenes |
| timm==0.6.5 | Modelos de Transformador de Visión |
| ptlflow==0.2.5 | Cálculo de flujo óptico (RAFT) |
| h5py | Almacenamiento/carga de características |
| imageio[ffmpeg] | Extracción de fotogramas de video |
| moviepy | Manipulación de video |
| scikit-learn, scipy | Utilidades ML |

---

### Paso 2: Modificar Módulo Transformador de PyTorch

⚠️ **Paso Crítico**: SAIS requiere salidas de mapa de atención que no están disponibles en PyTorch 1.8.0 estándar

```bash
# Navegar a su entorno conda
# Ruta de ejemplo: C:\Users\User\anaconda3\envs\SAIS\lib\python3.9\site-packages\torch\nn\modules\transformer.py

# Editar transformer.py:
# Línea 181: Agregar 'attn' como segunda salida de función mod
# Línea 294: Remover indexación [0] y agregar 'attn' como salida

# El método forward del transformador debería retornar (salida, pesos_atención)
```

**Instrucciones Detalladas**:
- Ubicar: `{anaconda_env}\lib\python3.9\site-packages\torch\nn\modules\transformer.py`
- Editar salida de MultiheadAttention para incluir pesos de atención
- Editar TransformerEncoderLayer/TransformerEncoder para propagar atención

---

### Paso 3: Descargar Pesos Pre-entrenados de DINO

```bash
# Crear directorio de salida
mkdir -p SAIS/scripts/dino-main/outputs

# Descargar pesos pre-entrenados de DINO (ViT-small-16)
# Opción A: Descarga manual
cd SAIS/scripts/dino-main/outputs
wget https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth
cd ../../..

# Opción B: Desde Python
import urllib.request
url = "https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
path = "SAIS/scripts/dino-main/outputs/dino_deitsmall16_pretrain.pth"
urllib.request.urlretrieve(url, path)
```

---

### Paso 4: Preparar Video de Entrada

```bash
# Crear directorio de videos si no existe
mkdir -p SAIS/videos

# Copiar su video quirúrgico en este directorio
# Ejemplo:
# SAIS/videos/surgery_001.mp4
# SAIS/videos/surgery_002.mp4

# Cada video debe tener nombre único
```

---

### Paso 5: Descargar/Preparar Parámetros del Modelo

Para inferencia, necesita parámetros del modelo entrenado:

```bash
# Estructura requerida:
# SAIS/
#   scripts/
#   params/
#     Fold_0/
#       params.zip          # Parámetros de arquitectura central
#       prototypes.zip      # Parámetros de prototipos para aprendizaje contrastivo
#     Fold_1/
#       ...

# Estos se obtienen después de entrenar en su conjunto de datos específico de video quirúrgico
# Codifican conocimiento específico de tarea (clases de gestos, patrones de habilidad, etc.)
```

**Si no tiene modelos entrenados**:
- Entrenar en su propio conjunto de datos (ver **Paso 8**)
- O obtener modelos pre-entrenados de los autores

---

### Paso 6: Ejecutar Pipeline de Inferencia (End-to-End)

#### Opción A: Script Bash Automatizado

```bash
# Navegar a raíz del repositorio
cd SAIS

# Ejecutar pipeline completo para un video
bash ./SAIS/main.sh -f surgery_001

# Qué hace main.sh (automáticamente):
# 1. video_to_frames.sh: Extraer fotogramas
# 2. generate_paths.py: Crear CSV de rutas
# 3. extract_representations.py (3x llamadas):
#    - Extracción de flujo óptico
#    - Extracción de características RGB  
#    - Extracción de características de flujo
# 4. run_experiments.py: Realizar inferencia
# 5. process_inference_results.py: Post-procesar predicciones
```

#### Opción B: Paso a Paso Manual

```bash
# 1. Extraer fotogramas del video
bash ./SAIS/scripts/video_to_frames.sh -f surgery_001

# 2. Generar asignaciones de rutas
python ./SAIS/scripts/generate_paths.py -f surgery_001 -p ./SAIS/

# 3. Extraer flujo óptico
python ./SAIS/scripts/extract_representations.py \
  --arch vit_small \
  --patch_size 16 \
  --model_type ViT_SelfSupervised_ImageNet \
  --batch_size_per_gpu 2 \
  --data_path ./SAIS/ \
  --data_list Custom \
  --save_type h5 \
  --optical_flow

# 4. Extraer características RGB (distribuido)
python -m torch.distributed.launch \
  ./SAIS/scripts/extract_representations.py \
  --arch vit_small \
  --patch_size 16 \
  --model_type ViT_SelfSupervised_ImageNet \
  --batch_size_per_gpu 1024 \
  --data_path ./SAIS/ \
  --data_list Custom \
  --save_type h5

# 5. Extraer características de flujo (distribuido)
python -m torch.distributed.launch \
  ./SAIS/scripts/extract_representations.py \
  --arch vit_small \
  --patch_size 16 \
  --model_type ViT_SelfSupervised_ImageNet \
  --batch_size_per_gpu 256 \
  --data_path ./SAIS/ \
  --data_list Custom \
  --save_type h5 \
  --optical_flow_to_reps

# 6. Realizar inferencia
python -m torch.distributed.launch \
  ./SAIS/scripts/run_experiments.py \
  -p ./SAIS/ \
  -data Custom_Gestures \
  -d Custom \
  -m ViT \
  -enc ViT_SelfSupervised_ImageNet \
  -t Prototypes \
  -mod RGB-Flow \
  -dim 384 \
  -bs 2 \
  -lr 1e-1 \
  -nc 2 \
  -bc \
  -sa \
  -domains in_vs_out \
  -ph Custom_inference \
  -dt reps \
  -e 1 \
  -f 1 \
  --inference

# 7. Procesar resultados
python ./SAIS/scripts/process_inference_results.py -p ./SAIS/
```

---

### Paso 7: Estructura del Directorio de Salida

Después del procesamiento, su directorio contendrá:

```
SAIS/
├── videos/                          # Videos de entrada
│   └── surgery_001.mp4
├── images/                          # NUEVO: Fotogramas extraídos
│   └── surgery_001/
│       ├── frame_00000.jpg
│       ├── frame_00001.jpg
│       └── ...
├── flows/                           # NUEVO: Mapas de flujo óptico
│   └── surgery_001/
│       ├── flow_00000.npy
│       └── ...
├── paths/                           # NUEVO: Asignaciones de rutas (CSV)
│   ├── Custom_FramePaths.csv
│   └── Custom_FlowPaths.csv
├── results/                         # NUEVO: Características extraídas (HDF5)
│   ├── rgb_features_custom.h5
│   └── flow_features_custom.h5
└── predictions/                     # NUEVO: Resultados de inferencia
    └── surgery_001_predictions.csv
```

---

### Paso 8: Entrenar Sus Propios Modelos

Si tiene datos de video quirúrgico etiquetados:

```bash
# Ejemplo: Entrenar clasificador de gestos
python -m torch.distributed.launch \
  ./SAIS/scripts/run_experiments.py \
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
  -domains gesture_classification \
  -ph train_val_test \
  -dt reps \
  -e 50 \
  -f 5 \
  -tf 1.0

# Parámetros explicados:
# -bs 8: Tamaño de lote (ajustar según memoria GPU)
# -lr 0.1: Velocidad de aprendizaje
# -nc 10: Número de clases de gesto
# -e 50: Entrenar por 50 épocas
# -f 5: Validación cruzada 5-fold
# -tf 1.0: Usar 100% de datos de entrenamiento
# Remover bandera --inference para modo entrenamiento
```

**Estructura requerida para entrenamiento**:
```
SAIS/
├── (preprocesamiento completado arriba)
├── annotations/
│   └── Custom_annotations.csv  # Columnas: video_name, frame_num, gesture_label
└── params/
    └── Fold_0/                 # Se creará después del entrenamiento
        ├── params.zip
        └── prototypes.zip
```

---

## 📈 Consideraciones de Desempeño

| Operación | Memoria GPU | Tiempo Est. | Notas |
|-----------|-----------|----------|-------|
| Extracción de características (video 10 min @ 20 FPS) | 8-12 GB | 2-5 min | Tamaño de lote afecta velocidad |
| Inferencia cada 10 min de video | 4 GB | 30 seg | Depende de complejidad del modelo |
| Entrenamiento (conjunto completo) | 16+ GB | Horas | Distribuido entre GPUs |
| Cálculo de flujo óptico | 6 GB | 3-10 min | Modelo RAFT es intensivo |

---

## 🔍 Parámetros de Configuración Clave

| Parámetro | Predeterminado | Rango | Impacto |
|-----------|---------|-------|--------|
| `snippet_length` | 5 | 1-30 | Ventana de contexto temporal (fotogramas) |
| `frame_skip` | 1 | 1-10 | Factor de submuestreo |
| `batch_size` | 8 | 1-256 | Velocidad de entrenamiento vs memoria GPU |
| `learning_rate` | 0.1 | 1e-4 a 1e-1 | Velocidad de convergencia y estabilidad |
| `rep_dim` | 384 | - | Dimensión de característica (fija para DINO) |
| `n_classes` | - | 2-∞ | Específico de tarea |

---

## 🐛 Problemas Comunes y Soluciones

| Problema | Causa | Solución |
|-------|-------|----------|
| "ModuleNotFoundError: No module named 'dino'" | DINO no en ruta | Asegurar `sys.path.append('./SAIS/scripts/dino-main')` en extract_representations.py |
| CUDA sin memoria | Tamaño de lote muy grande | Reducir parámetro `--batch_size_per_gpu` |
| "AttributeError: attention" | Transformador de PyTorch no modificado | Seguir Paso 2: Modificar transformer.py |
| Pesos de DINO faltantes | Archivo no descargado | Descargar de enlace FB Research (Paso 3) |
| No se generan predicciones | Parámetros del modelo faltantes | Asegurar `params/Fold_X/` existe con params.zip |
| Video no detectado | Estructura de directorio incorrecta | Colocar en `SAIS/videos/`, no subdirectorios |

---

## 📚 Gráfico de Dependencias de Componentes

```
run_experiments.py (Orquestador)
├── prepare_model.py
│   └── Transformador de Visión + Prototipos
├── prepare_dataset.py
│   ├── Archivos de características (HDF5)
│   └── Anotaciones (CSV)
├── train.py
│   ├── perform_training.py (single_epoch)
│   ├── prepare_miscellaneous.py (métricas)
│   └── prepare_dataset.py (DataLoader)
├── extract_representations.py
│   ├── DINO (dino-main/main_dino.py)
│   └── ptlflow (flujo óptico RAFT)
├── generate_paths.py (generación CSV)
│   └── Metadatos de video
└── process_inference_results.py
    └── Predicciones brutas → Salidas finales
```

---

## 📖 Referencias

- **Artículo**: ["A Vision Transformer for Decoding Surgery" - Nature Biomedical Engineering](https://www.nature.com/articles/s41551-023-01010-8)
- **DINO**: [Facebook Research - Emerging Properties in Self-Supervised ViTs](https://github.com/facebookresearch/dino)
- **ptlflow**: [PyTorch Lightning Optical Flow](https://github.com/hmorimitsu/ptlflow)
- **Transformadores de Visión**: [Artículo Original ViT - Google Research](https://arxiv.org/abs/2010.11929)

---

## 📝 Licencia

CC BY-NC 4.0 - Ver archivo LICENSE para detalles
