# Lista de Verificación de Configuración Local de SAIS y Guía Práctica

## ✅ Verificación Previa a la Configuración

Antes de comenzar, verifique que tiene:

- [ ] **GPU**: GPU NVIDIA con soporte CUDA (verificar: `nvidia-smi`)
- [ ] **Almacenamiento**: Al menos 100 GB de espacio libre (varía según tamaño de video)
- [ ] **RAM**: 16+ GB de RAM del sistema
- [ ] **Software**: Git, Python 3.9+, conda/pip
- [ ] **Video Quirúrgico**: Archivo MP4 listo

---

## 🔧 Instalación Paso a Paso

### Fase 1: Configuración del Entorno (30 min)

#### 1.1 Clonar Repositorio
```bash
git clone https://github.com/danikiyasseh/SAIS.git
cd SAIS
# Verificar estructura de directorios
ls -la
# Debería mostrar: README.md, requirements.txt, setup.py, SAIS/, LICENSE
```

#### 1.2 Crear Entorno Conda
```bash
# Crear con versión exacta de Python
conda create -n SAIS python=3.9.7 -y

# Activar entorno
conda activate SAIS

# Verificar activación (debería mostrar prefijo (SAIS) en terminal)
which python
# La salida debería contener: .../envs/SAIS/...
```

#### 1.3 Instalar Dependencias
```bash
# Instalar desde requirements.txt
pip install -r requirements.txt

# Verificar paquetes clave instalados
pip list | grep -E "torch|torchvision|opencv|h5py|ptlflow|timm"

# La salida esperada debería mostrar:
# torch                    1.8.0
# torchvision              0.9.0
# opencv-python            X.X.X
# h5py                     X.X.X
# ptlflow                  0.2.5
# timm                     0.6.5
```

#### 1.4 Instalar Paquete en Modo Desarrollo
```bash
pip install -e .

# Verificar instalación
python -c "import SAIS; print('Paquete SAIS instalado exitosamente')"
```

#### 1.5 Verificar Soporte CUDA de PyTorch
```bash
python -c "import torch; print(f'CUDA disponible: {torch.cuda.is_available()}'); print(f'Número de dispositivos: {torch.cuda.device_count()}')"

# Ejemplo de salida:
# CUDA disponible: True
# Número de dispositivos: 1
```

---

### Fase 2: Modificación del Transformador de PyTorch (20 min) ⚠️ CRÍTICO

#### 2.1 Ubicar transformer.py

```bash
# Encontrar su entorno conda
conda info -a | grep "active environment"

# Navegar al módulo transformador
# Ejemplo Windows:
# C:\Users\User\anaconda3\envs\SAIS\Lib\site-packages\torch\nn\modules\transformer.py

# Ejemplo Linux:
# ~/anaconda3/envs/SAIS/lib/python3.9/site-packages/torch/nn/modules/transformer.py

# Obtener la ruta exacta con Python:
python -c "import torch; import os; path = os.path.dirname(torch.__file__); print(os.path.join(path, 'nn/modules/transformer.py'))"
```

#### 2.2 Respaldar Archivo Original
```bash
# Windows PowerShell:
$TORCH_PATH = python -c "import torch; import os; print(os.path.dirname(torch.__file__))"
$TRANSFORMER_FILE = "$TORCH_PATH\nn\modules\transformer.py"
Copy-Item $TRANSFORMER_FILE "$TRANSFORMER_FILE.backup"

# Linux:
TORCH_PATH=$(python -c "import torch; import os; print(os.path.dirname(torch.__file__))")
TRANSFORMER_FILE="$TORCH_PATH/nn/modules/transformer.py"
cp $TRANSFORMER_FILE $TRANSFORMER_FILE.backup
```

#### 2.3 Editar Módulo Transformador

**Encontrar y modificar estas secciones**:

**Ubicación 1 - Alrededor de la línea 181 (método forward de atención multi-cabeza):**

Original:
```python
def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None,
            average_attn_weights=True):
    ...
    return attn_output, attn_output_weights
```

También debería retornar atención en la función llamadora.

**Ubicación 2 - Alrededor de la línea 294 (forward de TransformerEncoderLayer):**

Original:
```python
def forward(self, src, src_mask=None, src_key_padding_mask=None):
    ...
    return src[0]
```

Debería modificarse a:
```python
def forward(self, src, src_mask=None, src_key_padding_mask=None):
    ...
    return src[0], src[1]  # También retorna pesos de atención
```

**Recomendado**: Usar el repositorio oficial de SAIS como referencia para modificaciones exactas.

#### 2.4 Verificar Modificación
```bash
python -c "
import torch.nn as nn
import inspect

# Verificar si el transformador ahora retorna atención
transformer = nn.TransformerEncoderLayer(d_model=384, nhead=8)
source = inspect.getsource(transformer.forward)
if 'attn' in source:
    print('✓ Modificación del transformador parece exitosa')
else:
    print('⚠ Verificar nuevamente las modificaciones de transformer.py')
"
```

---

### Fase 3: Descargar Pesos Pre-entrenados (15 min)

#### 3.1 Crear Directorio de Salida
```bash
mkdir -p SAIS/scripts/dino-main/outputs
cd SAIS/scripts/dino-main/outputs
```

#### 3.2 Descargar Pesos de DINO

**Opción A: Usar wget (Linux/Mac/Windows con Git Bash)**
```bash
wget https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth \
  -O dino_deitsmall16_pretrain.pth

# Verificar descarga
ls -lh dino_deitsmall16_pretrain.pth
# Debería mostrar archivo ~350 MB
```

**Opción B: Usar PowerShell (Windows Nativo)**
```powershell
$url = "https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
$output = "dino_deitsmall16_pretrain.pth"
Invoke-WebRequest -Uri $url -OutFile $output

# Verificar
dir dino_deitsmall16_pretrain.pth
```

**Opción C: Usar Python**
```bash
python -c "
import urllib.request
import os

url = 'https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth'
output_dir = './SAIS/scripts/dino-main/outputs'
os.makedirs(output_dir, exist_ok=True)

filepath = os.path.join(output_dir, 'dino_deitsmall16_pretrain.pth')

print(f'Descargando pesos DINO a {filepath}...')
urllib.request.urlretrieve(url, filepath)
print(f'✓ Descargado exitosamente: {os.path.getsize(filepath) / 1e6:.1f} MB')
"
```

#### 3.3 Verificar Descarga
```bash
# Verificar que el archivo existe y su tamaño
ls -lh SAIS/scripts/dino-main/outputs/

# Debería mostrar:
# -rw-r--r-- 1 user group 350M ... dino_deitsmall16_pretrain.pth
```

---

### Fase 4: Preparar Video de Entrada (10 min)

#### 4.1 Crear Directorio de Videos
```bash
cd SAIS  # Volver a raíz del repositorio
mkdir -p SAIS/videos
```

#### 4.2 Agregar Su Video
```bash
# Copiar su video quirúrgico
# Windows:
Copy-Item "ruta/a/su/video.mp4" "SAIS/videos/video.mp4"

# Linux/Mac:
cp /ruta/a/su/video.mp4 SAIS/videos/video.mp4

# Verificar
ls -lh SAIS/videos/
```

#### 4.3 Requisitos de Video
- [ ] Formato: MP4 (códec H.264 preferido)
- [ ] Duración: Puede manejar videos de 5-60 min
- [ ] Velocidad de fotogramas: 20-25 FPS típico en videos quirúrgicos
- [ ] Resolución: 1080p o superior
- [ ] Tamaño: Esperar ~200-500 MB por 10 min

**Para verificar propiedades de video** (usando ffmpeg):
```bash
ffprobe -v error -select_streams v:0 \
  -show_entries stream=width,height,r_frame_rate,duration \
  -of default=noprint_wrappers=1:nokey=1:noescapes=1 \
  SAIS/videos/video.mp4
```

---

### Fase 5: Verificar Parámetros del Modelo Disponibles (10 min)

#### 5.1 Verificar Parámetros de Modelo Pre-entrenado
```bash
# Listar directorio de modelos
ls -la SAIS/params/

# Si está vacío o no existe, debe hacer:
# A) Entrenar el suyo (ver Fase 7)
# B) Descargar de autores
```

#### 5.2 Estructura Esperada
```bash
# Debería verse así eventualmente:
SAIS/params/
├── Fold_0/
│   ├── params.zip          # Pesos de arquitectura (~100-200 MB)
│   ├── prototypes.zip      # Vectores de prototipo (~50-100 MB)
│   └── checkpoint/         # Checkpoints opcionales de entrenamiento
├── Fold_1/
└── ...
```

#### 5.3 Obtener Modelos Pre-entrenados
```bash
# Si no está disponible, entrenar primero (Fase 7)
# O descargar de autores del artículo si se proporciona
# Verifique lanzamientos/problemas del repositorio para enlaces
```

---

### Fase 6: Prueba Rápida - Ejecutar Extracción de Características (15 min)

#### 6.1 Probar Extracción de Características de DINO

```bash
python -c "
import torch
import sys
sys.path.append('./SAIS/scripts/dino-main')
from main_dino import load_model

# Cargar DINO
model_name = 'dino_deitsmall16'
model = load_model(model_name, 'cpu')  # Usar CPU para prueba
print(f'✓ Cargado {model_name} exitosamente')

# Crear entrada ficticia
dummy_input = torch.randn(1, 3, 224, 224)

# Forward pass
with torch.no_grad():
    output = model(dummy_input)
    print(f'✓ Extracción de características funciona. Forma de salida: {output.shape}')
"
```

#### 6.2 Probar Lanzamiento Distribuido de PyTorch
```bash
# Prueba con 1 GPU
python -m torch.distributed.launch --nproc_per_node=1 \
  -c "print('✓ Lanzamiento distribuido funciona')"
```

---

## 🎬 Fase 7: Ejecutar Pipeline de Inferencia Completo

### 7.1 Prueba de Inferencia Rápida

```bash
# Activar entorno
conda activate SAIS

# Navegar a raíz del repositorio
cd SAIS

# Ejecutar pipeline end-to-end
bash ./SAIS/main.sh -f video

# Monitorear salida - debería mostrar:
# ✓ Extrayendo fotogramas...
# ✓ Generando rutas...
# ✓ Computando flujo óptico...
# ✓ Extrayendo características RGB...
# ✓ Extrayendo características de flujo...
# ✓ Ejecutando inferencia...
# ✓ Procesando resultados...
```

### 7.2 Salida Esperada

```bash
# Después de completarse, verificar salidas
ls -lh SAIS/

# Debería contener ahora:
# images/          # Fotogramas extraídos
# flows/           # Mapas de flujo óptico
# paths/           # Archivos CSV de rutas
# results/         # Archivos de características HDF5
# predictions/     # Predicciones finales
```

### 7.3 Ver Resultados

```bash
# Verificar predicciones
ls SAIS/predictions/

# Ver resultados CSV
head -20 SAIS/predictions/video_predictions.csv

# Formato esperado:
# video_name,action_id,gesture_label,confidence,frame_importance
# video,1,knot_tying,0.92,"[0.1,0.2,0.85,...]"
```

---

## 🏋️ Fase 8: Entrenar en Conjunto de Datos Personalizado (Opcional - 2+ horas)

### 8.1 Preparar Datos de Entrenamiento

```bash
# Estructura:
SAIS/
├── annotations/
│   └── train_annotations.csv
├── videos/
│   ├── surgery_001.mp4
│   ├── surgery_002.mp4
│   └── ...
```

### 8.2 Crear CSV de Anotaciones

**Formato** (train_annotations.csv):
```csv
video_name,frame_number,gesture_label,surgeon_id
surgery_001,0,rest,surgeon_a
surgery_001,1,rest,surgeon_a
surgery_001,25,knot_tying,surgeon_a
surgery_001,50,tissue_grasping,surgeon_a
surgery_001,75,suturing,surgeon_a
surgery_002,0,rest,surgeon_b
...
```

### 8.3 Extraer Características para Todos los Videos

```bash
# Procesar cada video (puede paralelizar)
for video in SAIS/videos/*.mp4; do
    videobase=$(basename "$video" .mp4)
    bash ./SAIS/main.sh -f "$videobase" --skip-inference
done
```

### 8.4 Entrenar Modelo

```bash
python -m torch.distributed.launch --nproc_per_node=1 \
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
  -domains gesture_classification \
  -ph train_val_test \
  -dt reps \
  -e 50 \
  -f 5 \
  -tf 1.0

# Monitorear registro de entrenamiento
tail -f ptlflow_logs/log_run.txt
```

### 8.5 Usar Modelo Entrenado para Inferencia

```bash
# Modelos automáticamente guardados en SAIS/params/Fold_X/
# Ahora la inferencia ejecuta con su modelo entrenado

bash ./SAIS/main.sh -f nuevo_video
```

---

## 📋 Lista de Verificación de Verificación

Después de completar todas las fases, verifique:

- [ ] **Entorno**: Prefijo `(SAIS)` en terminal
- [ ] **CUDA**: GPU disponible (`nvidia-smi` muestra dispositivos)
- [ ] **Paquetes**: `pip list | grep torch` muestra 1.8.0
- [ ] **DINO**: Archivo de pesos existe (~350 MB)
- [ ] **Transformador**: Modificado para retornar atención
- [ ] **Video**: Presente en `SAIS/videos/`
- [ ] **Params**: Archivos de modelo en `SAIS/params/Fold_0/` (si no entrena)
- [ ] **Prueba de ejecución**: Fase 7 completada exitosamente
- [ ] **Salida**: CSV de predicciones generado

---

## 🚀 Ejecutar Inferencia en Nuevos Videos (Plantilla de Repetición)

Una vez completada configuración, ejecutar en nuevos videos es simple:

```bash
# 1. Agregar video
cp nuevo_video_quirurgico.mp4 SAIS/videos/

# 2. Ejecutar pipeline
cd SAIS
bash main.sh -f nuevo_video_quirurgico

# 3. Verificar resultados
cat SAIS/predictions/nuevo_video_quirurgico_predictions.csv
```

---

## 🐛 Solución de Problemas Durante Configuración

### Problema: `ModuleNotFoundError: torch`

```bash
# Solución: Activar entorno conda
conda activate SAIS
python -c "import torch; print('OK')"
```

### Problema: `CUDA error: out of memory`

```bash
# Durante extracción de características, reducir tamaño de lote
# En main.sh, modificar:
python -m torch.distributed.launch ./SAIS/scripts/extract_representations.py \
  --batch_size_per_gpu 256  # Reducir de 1024 si hay OOM
```

### Problema: `FileNotFoundError: dino_deitsmall16_pretrain.pth`

```bash
# Verificar ruta
ls -la SAIS/scripts/dino-main/outputs/

# Descargar nuevamente si falta
cd SAIS/scripts/dino-main/outputs
wget https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth
```

### Problema: Fotogramas no extraídos

```bash
# Verificar formato de video
ffprobe -v error -select_streams v:0 SAIS/videos/video.mp4

# Verificar ffmpeg instalado
ffmpeg -version

# Intentar extracción manual
bash SAIS/scripts/video_to_frames.sh -f video
```

### Problema: Permiso denegado en scripts

```bash
# Hacer scripts ejecutables (Linux/Mac)
chmod +x SAIS/scripts/video_to_frames.sh
chmod +x SAIS/main.sh
```

---

## 📊 Monitoreo de Recursos del Sistema

### Monitorear Durante Procesamiento

**Windows PowerShell:**
```powershell
# Monitorear GPU
while($true) {
    nvidia-smi
    Start-Sleep -Seconds 1
}
```

**Linux/Mac:**
```bash
# Monitorear GPU
watch -n 1 nvidia-smi

# Monitorear uso de disco
watch -n 1 "du -sh SAIS/*"

# Monitorear memoria
watch -n 1 free -h
```

### Uso de Recursos Esperado

| Fase | Memoria GPU | CPU | Duración |
|-------|-----------|-----|----------|
| Extracción de características | 8-10 GB | 50-80% | 3-10 min / 10 min de video |
| Inferencia | 4 GB | 30-50% | 30-60 seg / 10 min de video |
| Época de entrenamiento | 12-16 GB | 70-90% | 10-30 min |

---

## ✨ Indicadores de Éxito

Después de ejecutar `bash main.sh`:

1. **Extracción de fotogramas comienza**:
   ```
   Extrayendo fotogramas de SAIS/videos/video.mp4...
   ```

2. **Cálculo de flujo óptico**:
   ```
   Cargando modelo RAFT...
   Procesando flujo óptico...
   Guardado: SAIS/flows/video/
   ```

3. **Extracción de características**:
   ```
   Cargando DINO ViT-small...
   Extrayendo características RGB...
   Extrayendo características de flujo...
   Guardado: SAIS/results/
   ```

4. **Inferencia**:
   ```
   Cargando modelo entrenado de: SAIS/params/Fold_0/
   Ejecutando inferencia en fragmentos...
   Predicciones guardadas...
   ```

5. **Resultados**:
   ```
   Procesando resultados...
   Predicciones finales: SAIS/predictions/video_predictions.csv
   ```

---

**Tiempo Total Estimado de Configuración**: 1-2 horas (depende de velocidades de descarga y GPU)

**Tiempo Estimado de Procesamiento por Video**: 5-15 minutos (video de 10 min en GPU moderna)
