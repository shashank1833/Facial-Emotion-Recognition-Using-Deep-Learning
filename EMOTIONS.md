# Emotion Classification Guide

This project classifies 7 distinct facial expressions. Below are the visual cues the model is trained to detect in each facial zone.

## 1. The 7 Emotion Classes

| Emotion | Primary Zones | Visual Key Cues |
| :--- | :--- | :--- |
| **Angry** | Brow, Mouth | Lowered brows, pressed lips, flared nostrils. |
| **Disgust** | Nose, Mouth | Wrinkled nose, raised upper lip, narrowed eyes. |
| **Fear** | Eyes, Brow | Raised brows (flat), widened eyes (sclera visible), open mouth. |
| **Happy** | Mouth, Eyes | Raised lip corners, "crow's feet" wrinkles around eyes. |
| **Sad** | Brow, Mouth | Inner brow corners raised, lip corners turned down. |
| **Surprise** | Brow, Eyes, Mouth | Curved/raised brows, wide eyes, dropped jaw. |
| **Neutral** | All | Absence of extreme muscle movement; baseline state. |

---

## 2. Dataset Distribution (FER-2013)
The model uses **Class Weights** to handle the following distribution in the training data:

- **Happy**: ~7,000 samples (Most common)
- **Neutral**: ~5,000 samples
- **Sad**: ~4,800 samples
- **Fear**: ~4,000 samples
- **Angry**: ~3,900 samples
- **Surprise**: ~3,100 samples
- **Disgust**: ~430 samples (Rarest - highly weighted in loss function)

---

## 3. How the Model "Sees"
1.  **Global View**: Analyzes the overall face shape and head tilt.
2.  **Local View (Zones)**:
    - `Mouth`: Detects smiles, frowns, or jaw drops.
    - `Eyes`: Detects widening or squinting.
    - `Forehead/Brow`: Detects furrowing or raising.
