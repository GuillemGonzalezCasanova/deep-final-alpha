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
