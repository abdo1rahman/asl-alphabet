# ASL Alphabet Recognition with a CNN (TensorFlow / Keras)

A convolutional neural network that classifies static American Sign Language (ASL) alphabet images into 29 classes, trained on the [ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) Kaggle dataset. The end goal is a **live webcam ASL transcriber using OpenCV** (planned, see the [Roadmap](#roadmap-live-transcriber-with-opencv)).

## TL;DR

- A small 64×64 CNN reaches **~100% test accuracy** on a randomly shuffled split of the dataset.
- Our first attempt split the data in file order and got **good validation accuracy (~88%) but poor test accuracy (49.3%)**.
- Investigating that gap showed the dataset is made of **four recording sessions per class** (different signers, backgrounds and lighting) stored back to back. Splitting in file order meant the test set was a whole session the model had never seen.
- Shuffling the data fixed the mismatch between train and test, but the resulting ~100% mostly measures how well the model knows these four sessions. The honest cross-session number is closer to **~46%**, so expect a gap when moving to a live webcam.
- Next step: load the saved `.keras` model in OpenCV, then collect your own webcam data to close the gap.

## Table of contents

1. [Dataset](#dataset)
2. [Model](#model)
3. [Experiments and results](#experiments-and-results)
4. [What the results mean](#what-the-results-mean)
5. [Using the saved model](#using-the-saved-model)
6. [Roadmap: live transcriber with OpenCV](#roadmap-live-transcriber-with-opencv)
7. [Known limitations](#known-limitations)
8. [Setup](#setup)

---

## Dataset

| Property | Value |
|---|---|
| Source | [Kaggle: grassknoted/asl-alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) |
| Training images | 87,000 |
| Classes | 29: `A`-`Z`, `del`, `nothing`, `space` |
| Images per class | 3,000 (perfectly balanced) |
| Image size | 200×200, RGB |
| Folder layout | `asl_alphabet_train/asl_alphabet_train/<class>/` (note the doubled folder name) |

### Data checks

A script scanned every training image. All 87,000 files are 200×200 RGB, every class has exactly 3,000 images, and no files are corrupt. No cleaning or resizing was needed, so images are only downscaled to 64×64 for training speed. Reading the whole dataset took about 8 minutes, so the pipeline decodes and resizes once and caches the result in memory.

### Issues found

1. **Near-duplicate frames.** Images look like frames extracted from short videos, so neighbouring files are almost identical.
2. **Four recording sessions per class.** Sample images from the start, middle and end of a class differ in skin tone, background and lighting. A plot of mean brightness against frame index showed sharp steps at the same indices in every class we checked:

   | Session | Frame index (per class) | Frames |
   |---|---|---|
   | S0 | 0-999 | 1,000 |
   | S1 | 1,000-1,999 | 1,000 |
   | S2 | 2,000-2,499 | 500 |
   | S3 | 2,500-2,999 | 500 |

3. **Limited diversity overall**, so models can latch onto session-specific cues such as background and lighting.
4. **`J` and `Z` involve motion** in real ASL but appear here as still frames.
5. **Visually similar letters** (e.g. `A`/`M`/`N`/`S`/`T`, `K`/`V`) are hard to separate at low resolution.
6. **The official test folder is tiny** (about one image per class), so it is not usable for evaluation. We create our own splits from the training data.

---

## Model

Input is 64×64×3. Augmentation layers are part of the model and are active only during training. Horizontal flips are deliberately avoided because they would change which hand makes the sign.

| Stage | Layers |
|---|---|
| Augmentation | `RandomRotation(0.05)`, `RandomZoom(0.1)`, `RandomTranslation(0.1, 0.1)`, `RandomBrightness(0.2)`, `RandomContrast(0.2)` |
| Preprocessing | `Rescaling(1/255)` (inside the model, so raw 0-255 pixels are fed in) |
| Conv block 1 | 2 × (`Conv2D(32, 3×3)` + `BatchNorm` + `ReLU`) → `MaxPooling2D` (64 → 32) |
| Conv block 2 | Same with 64 filters (32 → 16) |
| Conv block 3 | Same with 128 filters (16 → 8) |
| Head | `GlobalAveragePooling2D` → `Dropout(0.4)` → `Dense(29, softmax)` |

**Training setup**

- Optimizer: Adam, learning rate 1e-3; loss: `sparse_categorical_crossentropy`
- Batch size 64, up to 30 epochs
- Callbacks: `EarlyStopping(patience=5, restore_best_weights=True)`, `ReduceLROnPlateau(factor=0.5, patience=2)`, `ModelCheckpoint(save_best_only=True)`
- Data pipeline: `tf.data` with parallel decoding, caching (as `uint8` to save memory), shuffling (training only) and prefetching

---

## Experiments and results

### Experiment 1: contiguous split (first attempt, poor result)

Because the images are near-duplicate video frames, we tried to avoid leakage by splitting each class **in file order**: the first 70% of frames for training, the next 15% for validation and the last 15% for testing.

| Metric | Result |
|---|---|
| Train accuracy | ~99.9% |
| Validation accuracy | ~88% (best epoch 18, val loss 0.41; early stopping triggered at epoch 23) |
| **Test accuracy** | **49.3%** |

Validation looked reasonable, but the test result collapsed. The confusion matrix showed a few "catch-all" classes (`L`, `B`, `M`, `N`, `P`) absorbing many other letters. For example `L` had recall 0.99 but precision 0.19, and `W` and `Z` scored 0.00 recall.

### Diagnosis

Viewing the first, middle and last frames of the worst classes showed visibly different people, backgrounds and lighting. Plotting mean brightness against frame index then revealed the four sessions described above. File order is session order, so:

- the training set covered the first two sessions and part of the third,
- validation sat mostly inside the third session, right next to training frames,
- the test set was almost entirely the fourth session, which the model had never seen.

The 49% was therefore a real cross-session result, not a bug in the pipeline.

### Experiment 2: split by session

We split explicitly by session: train on S0 + S1, validate on S2, test on S3.

| Metric | Result |
|---|---|
| Validation accuracy | ~94.6% |
| **Test accuracy (unseen session S3)** | **45.6%** |

The failure pattern matched Experiment 1: `L`, `B`, `N`, `P` and `M` over-predicted, and `W` never predicted. Several letters (`U`, `V`, `X`, `Z`, `del`) had precision near 1.0 but recall near 0, meaning the model almost never predicted them on S3.

> **Note:** this run resumed training from the earlier model's weights rather than starting from a fresh model, so its validation figure is inflated. The test figure is unaffected, because S3 was never used for training. A clean rerun from fresh weights is recommended before drawing finer conclusions.

### Experiment 3: shuffled split (near-perfect result)

We then shuffled the data and used a random 70/15/15 split across all sessions.

| Metric | Result |
|---|---|
| **Test accuracy** | **~100%** (0.9999, loss 4.5e-4) |
| Per-class precision / recall / F1 | 1.00 for all 29 classes (450 test images per class) |

The confusion matrix is a clean diagonal. This is the model that was saved in `.keras` format.

### Results at a glance

| Experiment | How the data was split | Test accuracy | What the test set measures |
|---|---|---|---|
| 1. Contiguous | By file order | 49.3% | Generalisation to a mostly unseen session |
| 2. By session | Train S0+S1, val S2, test S3 | 45.6% | Generalisation to an unseen session |
| 3. Shuffled | Random across all sessions | ~100% | Recognition of the four known sessions |

---

## What the results mean

Shuffling resolved the mismatch between the training and test data: every split now contains frames from every session, so the model is no longer asked to handle conditions it never saw. That is why accuracy jumps from ~46-49% to ~100%.

It does not, by itself, make the model robust to new conditions. Because neighbouring frames are near-duplicates, a random split places almost identical images on both sides of the train/test boundary, so the ~100% largely reflects how well the model knows these four sessions. The session-held-out numbers (~46-49%) are the better estimate of how the model behaves with a new signer, background or camera.

**Implication for the live demo:** a webcam is effectively a fifth session. Expect noticeably lower accuracy than 100% at first, and plan to fine-tune with your own data (see the roadmap).

---

## Using the saved model

The saved model contains the rescaling and augmentation layers, so it expects raw pixel values in the 0-255 range. Class order must match training (alphabetical).

```python
import numpy as np
import tensorflow as tf

CLASS_NAMES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["del", "nothing", "space"]

model = tf.keras.models.load_model("asl_cnn.keras")  # use your saved file name

def predict(rgb_image_uint8):
    """rgb_image_uint8: HxWx3 RGB array, already resized to 64x64."""
    x = rgb_image_uint8.astype("float32")[None, ...]   # no /255, the model rescales
    probs = model(x, training=False).numpy()[0]
    return CLASS_NAMES[int(np.argmax(probs))], float(probs.max())
```

---

## Roadmap: live transcriber with OpenCV

Status: **planned**. A 64×64 CNN is small enough to run in real time on a laptop CPU.

1. **Run locally.** `cv2.VideoCapture(0)` cannot reach a webcam from a hosted notebook (Colab, Kaggle). Download the `.keras` file and run a script on your own machine, ideally with the same TensorFlow/Keras version used for training.
2. **Match the training preprocessing exactly:**
   - convert OpenCV's BGR frames to RGB (`cv2.COLOR_BGR2RGB`),
   - resize to 64×64 with `cv2.INTER_LINEAR`,
   - feed float32 values in the 0-255 range (no manual `/255`),
   - add a batch dimension.
3. **Locate the hand.** Training images are tight hand close-ups, so the whole webcam frame should not be fed in.
   - Start with a fixed on-screen ROI box that the user places their hand in.
   - Upgrade later to a hand detector such as MediaPipe Hands to crop automatically (check Python version support first).
4. **Smooth predictions.** Keep the last ~10 predictions and only accept a letter if it is the majority and its mean confidence passes a threshold (e.g. 0.8).
5. **Turn predictions into a transcript:**
   - commit a letter after it has been stable for about a second,
   - `space` appends a space and `del` removes the last character,
   - `nothing` resets the lock so repeated letters (e.g. "LL") can be entered by dropping the hand out of frame in between,
   - overlay the prediction, confidence and transcript with `cv2.putText`.
6. **Close the domain gap with your own data.** Add a capture mode that saves ROI crops per letter on a key press. Record 100-200 varied frames per letter across several separate sessions (different lighting, positions and backgrounds), fine-tune on the original dataset plus the new frames with stronger colour/brightness augmentation, and hold out one of your own sessions as the honest test set.
7. **Further improvements if needed:**
   - transfer learning (e.g. MobileNetV2) at 96-128 px,
   - hue/saturation jitter and random grayscale augmentation,
   - leave-one-session-out cross-validation for a more reliable generalisation estimate,
   - TFLite conversion if inference speed becomes an issue.

---

## Known limitations

- Accuracy on new signers, lighting or cameras is expected to be much lower than the ~100% shuffled-split figure.
- `J` and `Z` are motion gestures and are only represented by single frames.
- Similar handshapes (`A`/`M`/`N`/`S`/`T`, `K`/`V`) are hard to tell apart at 64×64.
- The model classifies one image at a time; it does not model sequences, so it cannot capture motion.
- The dataset has no signer or session labels. Session boundaries were inferred from brightness changes and verified visually on a sample of classes.

---

## Setup

```bash
pip install tensorflow numpy scikit-learn matplotlib pillow
# for the live transcriber (planned):
pip install opencv-python
```

1. Download the dataset from Kaggle and extract it so the training images are at `asl_alphabet_train/asl_alphabet_train/<class>/`.
2. Set `ROOT` in the training code to that path.
3. Train, evaluate and save the model as a `.keras` file.

## Credits

- Dataset: [ASL Alphabet by grassknoted](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) on Kaggle.
