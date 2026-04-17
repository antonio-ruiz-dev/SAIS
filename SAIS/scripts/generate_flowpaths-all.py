"""
Genera CSVs de FlowPaths con columnas path1, path2, label
a partir de los CSVs de gestos de JIGSAWS.

Uso:
    python generate_flowpaths_csv.py

Salida:
    SAIS/paths/JIGSAWS_Suturing_FlowPaths.csv
    SAIS/paths/JIGSAWS_Knot_Tying_FlowPaths.csv
    SAIS/paths/JIGSAWS_Needle_Passing_FlowPaths.csv
"""

import os
import pandas as pd

# ── Configuración ────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # raíz del proyecto
IMAGES_DIR  = os.path.join(BASE_DIR, 'SAIS', 'images')   # donde están los frames .jpg
PATHS_DIR   = os.path.join(BASE_DIR, 'SAIS', 'paths')    # donde están los CSVs de gestos

TASKS = ['Suturing', 'Knot_Tying', 'Needle_Passing']

# Mapeo SubjectNN → letra  (Subject02→B, Subject03→C, ...)
def subject_to_letter(subject_str):
    """'Subject02' → 'B'"""
    num = int(subject_str.replace('Subject', ''))
    return chr(ord('A') + num - 1)   # 1→A, 2→B, 3→C ...

# Número de trial  'Trial001' → '001'
def trial_to_num(trial_str):
    return trial_str.replace('Trial', '')

# ── Función principal ────────────────────────────────────────────────────────
def generate_flowpaths(task):
    gesture_csv = os.path.join(PATHS_DIR, f'JIGSAWS_{task}_FlowPaths.csv')
    output_csv  = os.path.join(PATHS_DIR, f'JIGSAWS_{task}_FlowPaths_v2.csv')
    print(f'Generando FlowPaths para {task} desde {gesture_csv} → {output_csv}...')

    if not os.path.exists(gesture_csv):
        print(f'[SKIP] No encontrado: {gesture_csv}')
        return

    df = pd.read_csv(gesture_csv)
    rows = []

    for _, row in df.iterrows():
        subject = subject_to_letter(row['Subject'])
        trial   = trial_to_num(row['SuperTrial'])
        gesture = row['Gesture']
        start   = int(row['StartFrame'])
        end     = int(row['EndFrame'])
        label   = gesture  # etiqueta = gesto (G1, G2, ...)

        # Nombre de la carpeta de frames: e.g. Suturing_B001
        folder_name = f'{task}_{subject}{trial}'
        folder_path = os.path.join(IMAGES_DIR, folder_name)
        print(f'Procesando {folder_name} ({gesture}: frames {start}-{end})...') 
        if not os.path.exists(folder_path):
            print(f'[WARN] Carpeta no encontrada: {folder_path}')
            continue

        # Generar pares consecutivos dentro del segmento gestual
        for frame_num in range(start, end):  # end-1 es el último frame válido como path1
            frame1_name = f'frames_{frame_num:08d}.jpg'
            frame2_name = f'frames_{frame_num + 1:08d}.jpg'

            frame1_full = os.path.join(folder_path, frame1_name)
            frame2_full = os.path.join(folder_path, frame2_name)

            # Solo agregar si ambos frames existen
            if not os.path.exists(frame1_full):
                continue
            if not os.path.exists(frame2_full):
                continue

            # Rutas relativas a BASE_DIR (como las espera extract_representations.py)
            path1_rel = os.path.relpath(frame1_full, BASE_DIR).replace('\\', '/')
            path2_rel = os.path.relpath(frame2_full, BASE_DIR).replace('\\', '/')
            
            rows.append({
                'path1':  path1_rel,
                'path2':  path2_rel,
                'label':  label,
                'folder': folder_name,
                'frame':  frame_num,
            })

    if not rows:
        print(f'[{task}] Sin filas generadas. Verifica rutas y nombres de carpetas.')
        return

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_csv, index=False)
    print(f'[{task}] {len(out_df)} pares generados → {output_csv}')

# ── Entry point ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    for task in TASKS:
        generate_flowpaths(task)
