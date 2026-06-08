# CNNbaseline + augmentation — Cómo ejecutarlo

Entrena la misma CNN baseline dos veces (con y sin data augmentation) y evalúa
ambas para poder compararlas.

## 1. Preparación

1. Descarga el dataset *YouTube Faces with Keypoints* de Kaggle y coloca los
   archivos `.npz` en `Data/faces_with_keypoints/` (junto con
   `Data/youtube_faces_with_keypoints_full.csv`).
2. Ejecuta **todas** las celdas de `preprocess.ipynb` para generar
   `Data/preprocessed/images_50k.npy` y `keypoints_50k.npy`.

## 2. Lo que TIENES que cambiar

Cada notebook tiene una ruta absoluta hardcodeada al principio:

```python
DATA_DIR = r'C:\Users\guill\...\deep-final-alpha\Data'
```

Edita `DATA_DIR` en cada notebook para que apunte a **tu** carpeta `Data/`. No hace falta cambiar nada más.

## 3. Ejecutar

Ejecuta cada notebook de arriba a abajo, en este orden:

```
CNNbaseline + augmentation/
├── baseline/train_BaselineCNN.ipynb          # 1. entrenar (sin augmentation)
├── baseline/evaluate.ipynb                   # 2. evaluar
├── baseline_augmented/train_baseline_augmented.ipynb  # 3. entrenar (con augmentation)
└── baseline_augmented/evaluate.ipynb         # 4. evaluar
```

Cada rama guarda archivos distintos (`best_baseline_cnn.pth` vs
`best_baseline_augmented.pth`), así que no se pisan entre sí. Compara el error de
validación que reportan los dos `evaluate.ipynb`.


# NaimishNet — Cómo ejecutarlo

Entrena NaimishNet en cinco configuraciones distintas (50k/100k frames, lr=1e-3/1e-4)
y evalúa los resultados comparándolos con la Baseline CNN.

## 1. Preparación
1. Descarga el dataset *YouTube Faces with Keypoints* de Kaggle y coloca los
   archivos `.npz` en `Data/faces_with_keypoints/`.
2. Ejecuta **todas** las celdas de `preprocess.ipynb` para generar
   `Data/preprocessed/images_50k.npy` y `keypoints_50k.npy`.
3. Ejecuta **todas** las celdas de `preprocess_100k.ipynb` para generar
   `Data/preprocessed/images_100k.npy` y `keypoints_100k.npy`.
**IMPORTANTE**
Si no quieres ejecutar el preprocess y el training y solo quieres ejecutar el test.ipynb y error_analysis.ipynb, este link del drive tiene un zip con todos los .pkl, .npy.
Simplemente descomprimes los archivos en la carpeta preprocessed y ejecutas los dos test.ipynb y error_analysis.ipynb.

## 2. Lo que TIENES que cambiar
Cada notebook tiene una ruta absoluta hardcodeada al principio:
```python
DATA_DIR = r'C:\Users\alber\...\deep-final-alpha\Data'
```
Edita `DATA_DIR` en cada notebook para que apunte a **tu** carpeta `Data/`. No hace falta cambiar nada más.

## 3. Ejecutar
Ejecuta cada notebook de arriba a abajo, en este orden:

```
NaimishNet/
├── preprocess.ipynb                  # 1. preprocesar 50k frames (si no se hizo antes)
├── preprocess_100k.ipynb             # 2. preprocesar 100k frames (si no se hizo antes)
├── baseline/train_BaselineCNN.ipynb  # 3. entrenar (sin augmentation)
├── train_naimishnet.ipynb            # 4. entrenar las 4 variantes de NaimishNet
│                                     #    (50k lr=1e-3, 100k lr=1e-3,
│                                     #     50k lr=1e-4, 100k lr=1e-4)
├── test.ipynb                        # 5. evaluar sobre 10k frames nunca vistos
└── error_analysis.ipynb              # 6. análisis de errores y patrones de fallo
```
## 4. Archivos generados
Tras ejecutar `train_naimishnet.ipynb` encontrarás en `Data/`:

| Archivo | Descripción |
|---|---|
| `naimishnet_50k.pth` | Mejor checkpoint — 50k frames, lr=1e-3 |
| `naimishnet_100k.pth` | Mejor checkpoint — 100k frames, lr=1e-3 |
| `naimishnet_50k_4.pth` | Mejor checkpoint — 50k frames, lr=1e-4 |
| `naimishnet_100k_4.pth` | Mejor checkpoint — 100k frames, lr=1e-4 |
| `losses_naimish_50k.pkl` | Curvas de loss — 50k, lr=1e-3 |
| `losses_naimish_100k.pkl` | Curvas de loss — 100k, lr=1e-3 |
| `losses_naimish_50k_4.pkl` | Curvas de loss — 50k, lr=1e-4 |
| `losses_naimish_100k_4.pkl` | Curvas de loss — 100k, lr=1e-4 |

## 5. Notas
- Los pasos 1 y 2 se pueden omitir si los `.npy` ya existen de una ejecución anterior.
- `train_naimishnet.ipynb` entrena las 4 variantes secuencialmente — puede tardar varias horas en CPU. Se recomienda GPU.
- `test.ipynb` y `error_analysis.ipynb` requieren que existan los 4 `.pth` y que `frame_index.pkl` esté en `Data/preprocessed/`.
