# Lab 4: Clinical Vision — Chest X-Ray Pneumonia Detection

## A Plain-Language Guide to Every Concept in the Code

---

## Table of Contents

1. [What Are We Building?](#1-what-are-we-building)
2. [The Big Picture: How Computers Learn to See](#2-the-big-picture-how-computers-learn-to-see)
3. [Cell-by-Cell Walkthrough](#3-cell-by-cell-walkthrough)
   - [Cell 1 — Loading the Dataset](#cell-1--loading-the-dataset)
   - [Cell 2 — Preprocessing the Images](#cell-2--preprocessing-the-images)
   - [Cell 3 — Loading a Pre-trained Model](#cell-3--loading-a-pre-trained-model)
   - [Cell 4 — Testing Before Fine-Tuning (The "Fail")](#cell-4--testing-before-fine-tuning-the-fail)
   - [Cell 5 — Modifying the Model for Our Task](#cell-5--modifying-the-model-for-our-task)
   - [Cell 6 — Fine-Tuning (The Actual Training)](#cell-6--fine-tuning-the-actual-training)
   - [Cell 7 — Evaluation with Clinical Metrics](#cell-7--evaluation-with-clinical-metrics)
   - [Cell 8 — Before vs. After Comparison](#cell-8--before-vs-after-comparison)
   - [Stretch Challenge — Unfreezing Feature Layers](#stretch-challenge--unfreezing-feature-layers)
4. [Key Concepts Glossary](#4-key-concepts-glossary)
5. [Clinical Takeaways](#5-clinical-takeaways)

---

## 1. What Are We Building?

Imagine a hospital that processes hundreds of chest X-rays every day. A doctor has to look at each one and decide: does this patient have pneumonia or not? That takes time, and some urgent cases might wait in a queue.

In this lab, we teach a computer to look at chest X-rays and flag the ones that likely show pneumonia. The computer won't replace the doctor — it acts like a fast first filter that says "hey, this one looks suspicious, check it soon."

We do this in three stages:

1. **Start with a model that already knows how to "see"** (it was trained on millions of everyday photos — cats, cars, food).
2. **Show it that it knows nothing about medicine** (it thinks X-rays are hourglasses and gowns).
3. **Teach it the difference between normal lungs and pneumonia** using labeled X-ray images.

---

## 2. The Big Picture: How Computers Learn to See

### What Is a Neural Network?

Think of a neural network as a giant stack of filters. Each filter looks for a specific visual pattern:

- Early filters detect simple things: edges, corners, brightness changes.
- Middle filters combine those into shapes: circles, textures, curves.
- Late filters recognize complex objects: a face, a wheel, a rib cage.

When you feed an image into the network, it passes through all these filters, and the network outputs a prediction like "I'm 85% sure this is a cat."

### What Is a Convolutional Neural Network (CNN)?

A CNN is a neural network specifically designed for images. The word "convolutional" refers to how it slides small filters across the image, checking every region. It's like looking at an image through a tiny magnifying glass that moves across every patch, recording what patterns it finds.

### What Is Transfer Learning?

Training a CNN from scratch requires millions of images and days of computing time. Transfer learning is a shortcut:

1. Take a model someone already trained on a huge dataset (ImageNet — 14 million photos of everyday objects).
2. Keep the early and middle filters (edges, shapes, textures — these are universal).
3. Replace only the final decision layer to match your new task.

It's like hiring an experienced photographer and saying: "You already know how to analyze images — now I just need you to learn the difference between healthy lungs and infected lungs."

### What Is a GPU and Why Do We Need One?

A GPU (Graphics Processing Unit) is a chip originally designed for video games. It turns out GPUs are also great at the math behind neural networks because they can do thousands of calculations at the same time (in parallel). Training our model on a CPU would take hours; on a GPU, it takes about 4–5 minutes.

---

## 3. Cell-by-Cell Walkthrough

### Cell 1 — Loading the Dataset

```python
ds = load_dataset("hf-vision/chest-xray-pneumonia")
```

**What's happening:** We download a public dataset of 5,856 chest X-ray images from Hugging Face, a platform that hosts datasets and AI models for free.

**The dataset has three splits:**

| Split | Purpose | Size |
|---|---|---|
| Train | Images the model learns from | 5,216 |
| Validation | Images used to check progress during training | 16 |
| Test | Images the model has never seen — used for the final grade | 624 |

**Class imbalance:** There are about 3× more pneumonia images (3,875) than normal images (1,341) in the training set. This matters because the model could learn to just always say "pneumonia" and still get a decent accuracy score. We'll address how to interpret results fairly in the evaluation section.

**The visualization** at the end of this cell displays 8 X-rays in a grid — 4 normal on top, 4 pneumonia on the bottom — so you can try to spot the difference yourself. (Spoiler: it's hard even for humans without training.)

---

### Cell 2 — Preprocessing the Images

Before feeding images into the model, we need to standardize them. Think of it like formatting an essay before submitting it — same content, but now it fits the expected template.

#### Step 1: Resize

```python
transforms.Resize((224, 224))
```

MobileNetV2 expects images that are exactly 224 × 224 pixels. Real X-rays come in all shapes and sizes, so we resize them all to this fixed dimension.

#### Step 2: Convert to a Tensor

```python
transforms.ToTensor()
```

A "tensor" is just a multi-dimensional array of numbers. A color image becomes a 3D tensor with shape `[3, 224, 224]`:

- **3** = three color channels (Red, Green, Blue)
- **224 × 224** = the pixel grid

Each value is a number between 0 and 1 representing brightness.

#### Step 3: Normalize

```python
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

Normalization shifts and scales pixel values so they match what MobileNetV2 was trained on. These specific numbers are the average color values of the ImageNet dataset. Without this step, the model's internal filters would receive inputs in an unexpected range and perform poorly.

**Analogy:** If a scale was calibrated using kilograms, you wouldn't feed it values in pounds and expect accurate results. Normalization converts our data into the "units" the model was calibrated for.

#### The Dataset Class

```python
class CXRDataset(Dataset):
```

This custom class acts as a bridge between the Hugging Face dataset format and PyTorch's DataLoader. It does three things for each image:

1. Converts it to RGB (3 channels) — some X-rays are grayscale (1 channel) and the model expects 3.
2. Applies the transforms above.
3. Returns the processed image paired with its label (0 = Normal, 1 = Pneumonia).

#### The DataLoader

```python
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
```

Instead of feeding images one at a time, we group them into **batches** of 32. This is faster because the GPU can process 32 images simultaneously. The `shuffle=True` randomizes the order each time so the model doesn't accidentally memorize a sequence.

---

### Cell 3 — Loading a Pre-trained Model

```python
model = models.mobilenet_v2(pretrained=True)
```

**MobileNetV2** is a lightweight CNN designed by Google. "Mobile" means it's efficient enough to run on a phone. Despite being small (only ~3.4 million parameters), it's surprisingly good at recognizing images.

**What does `pretrained=True` mean?** It downloads a version of MobileNetV2 that was already trained on ImageNet — a dataset of 14 million images across 1,000 categories (dogs, cars, mushrooms, pizza, etc.). The model already knows how to extract useful visual features from images.

**The classifier layer:**

```
Sequential(
  (0): Dropout(p=0.2)
  (1): Linear(in_features=1280, out_features=1000)
)
```

This is the model's "decision head." It takes 1,280 features from the last convolutional layer and maps them to 1,000 categories. We'll need to change this because we only have 2 categories (Normal vs. Pneumonia).

**What is Dropout?** Dropout randomly "turns off" 20% of neurons during training. This forces the model to not rely on any single neuron, making it more robust — like studying with different groups of friends so you don't depend on just one person for answers.

---

### Cell 4 — Testing Before Fine-Tuning (The "Fail")

This cell is the "aha moment" of the lab. We show that a model trained on everyday photos has absolutely no idea what a chest X-ray is.

```
X-ray #1 (True: NORMAL) -> [('gown', '21.9%'), ('oxygen mask', '15.0%'), ('hoop skirt', '7.9%')]
X-ray #3 (True: NORMAL) -> [('hourglass', '32.6%'), ...]
X-ray #4 (True: NORMAL) -> [('isopod', '46.6%'), ('jellyfish', '15.9%'), ...]
```

The model thinks chest X-rays look like gowns, hourglasses, and jellyfish! It has never seen a medical image before, so it's guessing from the 1,000 categories it knows.

**Key concept — `torch.softmax`:** The raw output of the model is a list of 1,000 numbers (called "logits"). Softmax converts these into probabilities that add up to 100%. The highest probability is the model's best guess.

**Key concept — `torch.no_grad()`:** During evaluation, we don't need to compute gradients (the math used for learning). Wrapping code in `torch.no_grad()` tells PyTorch to skip that computation, which saves memory and speeds things up.

**Why is baseline accuracy ~50%?** With a 1,000-class model trying to do a 2-class task, the predictions are essentially random. Random guessing on a binary task gives roughly 50% accuracy.

---

### Cell 5 — Modifying the Model for Our Task

This is where we perform the "surgery" on the model.

#### Replace the Final Layer

```python
model.classifier[1] = nn.Linear(model.last_channel, 2)
```

We swap out the 1,000-class output layer and replace it with a 2-class layer (`nn.Linear(1280, 2)`). Now the model outputs two numbers: one score for "Normal" and one for "Pneumonia."

#### Freeze the Feature Extractor

```python
for param in model.features.parameters():
    param.requires_grad = False
```

"Freezing" means we lock the existing filters in place — they won't change during training. Only the new classifier layer will learn.

**Why freeze?** Two reasons:

1. **The features are already good.** Edge detectors, shape recognizers, and texture analyzers work just as well on X-rays as on photos. There's no need to relearn them.
2. **We have limited data.** With only 5,216 training images, updating millions of parameters would risk overfitting (memorizing the training data instead of learning general patterns).

**Result:** Only 2,562 out of 2,226,434 parameters are trainable — that's 0.1% of the model. We're training a tiny sliver while leveraging all the knowledge already baked into the frozen layers.

---

### Cell 6 — Fine-Tuning (The Actual Training)

This is where the model actually learns to detect pneumonia.

#### Loss Function

```python
criterion = nn.CrossEntropyLoss()
```

Cross-entropy loss measures how wrong the model's predictions are. If the model is confident and correct, loss is low. If it's confident and wrong, loss is high. The goal of training is to minimize this number.

**Analogy:** Imagine a student taking a multiple-choice test and writing down how confident they are for each answer. Cross-entropy penalizes the student harshly for being confidently wrong, and rewards them for being confidently right.

#### Optimizer

```python
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
```

The optimizer is the algorithm that adjusts the model's weights to reduce the loss. **Adam** is a popular optimizer that adapts its step size for each parameter automatically.

**Learning rate (`lr=0.001`)** controls how big each adjustment step is:

- Too high → the model overshoots and never settles on good weights (like trying to park by flooring the gas pedal).
- Too low → training takes forever and might get stuck.
- 0.001 is a common starting point that works well for many tasks.

#### The Training Loop

For each epoch (one full pass through the training data):

1. **Forward pass:** Feed a batch of 32 images through the model, get predictions.
2. **Compute loss:** Compare predictions to the true labels.
3. **Backward pass (`loss.backward()`):** Calculate how each weight contributed to the error (this is called backpropagation).
4. **Update weights (`optimizer.step()`):** Nudge the weights slightly to reduce the error.
5. **Repeat** for every batch in the training set.

**What is an Epoch?** One epoch = the model sees every training image once. We train for 3 epochs, meaning the model cycles through the entire dataset 3 times.

**Training results:**

| Epoch | Loss | Accuracy |
|---|---|---|
| 1 | 0.2677 | 89.3% |
| 2 | 0.1645 | 93.1% |
| 3 | 0.1501 | 93.8% |

Loss goes down and accuracy goes up — the model is learning. Total training time: about 4.5 minutes on a T4 GPU.

---

### Cell 7 — Evaluation with Clinical Metrics

Now we test the model on 624 images it has never seen before.

#### Accuracy

```
Overall Test Accuracy: 86.1%
```

The model gets 86.1% of predictions correct. But accuracy alone can be misleading — especially with imbalanced data. Let's look at more detailed metrics.

#### Precision, Recall, and F1-Score

These three metrics tell different stories about the model's mistakes:

**Precision** — "When the model says pneumonia, how often is it right?"

```
Pneumonia precision: 0.82 (82%)
```

Out of every 100 images the model flags as pneumonia, 82 actually have pneumonia. The other 18 are false alarms (normal lungs incorrectly labeled as pneumonia).

**Recall (Sensitivity)** — "Out of all actual pneumonia cases, how many did the model catch?"

```
Pneumonia recall: 0.99 (99%)
```

Out of 390 actual pneumonia cases, the model correctly identified 385. It only missed 5. This is very important in medicine — missing a sick patient is far worse than a false alarm.

**F1-Score** — The harmonic mean of precision and recall. It balances both into a single number. A perfect F1 is 1.0; our model achieves 0.90 for pneumonia.

#### The Confusion Matrix

```
                    Predicted NORMAL  Predicted PNEUMONIA
Actual NORMAL              152                  82
Actual PNEUMONIA             5                 385
```

Reading this table:

- **True Negatives (152):** Normal lungs correctly identified as normal.
- **False Positives (82):** Normal lungs incorrectly flagged as pneumonia. These patients would get an unnecessary follow-up test — inconvenient but not harmful.
- **False Negatives (5):** Pneumonia cases the model missed. These are the dangerous errors — a sick patient might not get treated.
- **True Positives (385):** Pneumonia correctly identified. The model caught almost all of them.

#### Why Recall Matters More in Screening

In a screening tool, we'd rather have false alarms (false positives) than missed cases (false negatives). A false alarm leads to an extra test; a missed case could lead to untreated pneumonia. Our model has 99% recall — it catches nearly every pneumonia case, which is exactly what we want for a screening tool.

---

### Cell 8 — Before vs. After Comparison

This cell brings everything together visually:

**Summary table:**

| Metric | Before (ImageNet) | After (Fine-tuned) |
|---|---|---|
| Accuracy | ~50% (random guessing) | 86.1% |
| Pneumonia Recall | ~0% (no concept of pneumonia) | 99% |
| Training Time | N/A | ~4.5 minutes |

**Visual predictions:** Six test X-rays are displayed with the model's prediction, confidence score, and whether it was correct. You can see the model is confident and usually right.

**Training curves:** Two plots show loss decreasing and accuracy increasing over the 3 epochs — visual proof that the model is learning.

---

### Stretch Challenge — Unfreezing Feature Layers

In the main lab, we only trained the final classifier (0.1% of the model). In the stretch challenge, we **unfreeze** the last feature extraction block too, allowing it to adapt to X-ray-specific patterns.

```python
for param in model.features[17:].parameters():
    param.requires_grad = True
```

Now 39.9% of parameters are trainable. We use **differential learning rates**:

```python
optimizer = optim.Adam([
    {'params': model.features[17:].parameters(), 'lr': 1e-4},   # slow
    {'params': model.classifier.parameters(), 'lr': 1e-3},       # fast
])
```

The feature layers use a 10× smaller learning rate (0.0001 vs 0.001) because we want to gently adjust existing knowledge, not overwrite it.

**Results:** Training accuracy improved to 97.1%, but test accuracy actually dropped to 81.6%. This is a classic case of **overfitting** — the model memorized the training data so well that it became worse at generalizing to new images. With only 5,216 training images, unfreezing too many layers gives the model too much freedom and not enough data to learn from.

---

## 4. Key Concepts Glossary

| Term | Plain-Language Definition |
|---|---|
| **Backpropagation** | The algorithm that figures out how much each weight contributed to the error, so we know how to adjust them. Like tracing back through a recipe to find which step made the dish too salty. |
| **Batch** | A group of images processed together (32 in this lab). Processing in batches is faster than one-by-one. |
| **Batch Size** | How many images are in each group. Larger batches train faster but use more GPU memory. |
| **CNN** | Convolutional Neural Network — a type of neural network designed for images that slides filters across the image to detect patterns. |
| **Confusion Matrix** | A 2×2 table showing what the model got right and wrong, broken down by each class. |
| **Cross-Entropy Loss** | A number measuring how wrong the model's predictions are. Lower is better. |
| **DataLoader** | A PyTorch tool that feeds batches of data to the model during training. |
| **Dropout** | Randomly disabling some neurons during training to prevent over-reliance on any one neuron. |
| **Epoch** | One complete pass through the entire training dataset. |
| **F1-Score** | A single number balancing precision and recall. Ranges from 0 (worst) to 1 (best). |
| **False Negative** | A sick patient the model missed (said "normal" when it's actually pneumonia). |
| **False Positive** | A healthy patient the model incorrectly flagged (said "pneumonia" when lungs are normal). |
| **Feature Extractor** | The convolutional layers of the model that detect visual patterns (edges, shapes, textures). |
| **Fine-Tuning** | Taking a pre-trained model and training it further on a new, specific dataset. |
| **Freezing** | Locking model weights so they don't change during training. |
| **GPU** | Graphics Processing Unit — hardware that speeds up neural network training by doing many calculations in parallel. |
| **Gradient** | The direction and magnitude of change needed to reduce the loss. Computed during backpropagation. |
| **ImageNet** | A massive dataset of 14 million labeled images across 1,000 categories, used to pre-train many vision models. |
| **Learning Rate** | How big each weight adjustment step is during training. Too high = unstable; too low = slow. |
| **Logits** | The raw output numbers from the model before they are converted to probabilities. |
| **MobileNetV2** | A lightweight CNN designed by Google, efficient enough to run on mobile phones. |
| **Normalization** | Scaling pixel values to match the range the model was trained on. |
| **Optimizer (Adam)** | The algorithm that adjusts model weights to reduce loss. Adam adapts its step size automatically. |
| **Overfitting** | When the model memorizes training data instead of learning general patterns, causing poor performance on new data. |
| **Parameters** | The numbers (weights) inside the model that get adjusted during training. MobileNetV2 has ~2.2 million. |
| **Precision** | Of all the cases the model flagged as positive, what fraction actually were positive? |
| **Pre-trained Model** | A model that was already trained on a large dataset and can be reused for new tasks. |
| **Recall (Sensitivity)** | Of all the actual positive cases, what fraction did the model catch? |
| **Softmax** | A function that converts raw model outputs into probabilities that add up to 1 (100%). |
| **Tensor** | A multi-dimensional array of numbers — the data structure neural networks operate on. |
| **Transfer Learning** | Reusing a model trained on one task (e.g., recognizing everyday objects) for a different task (e.g., detecting pneumonia). |

---

## 5. Clinical Takeaways

1. **AI is a tool, not a replacement.** This model acts as a screening filter — it flags suspicious X-rays for a radiologist to review, not to make final diagnoses.

2. **Transfer learning is powerful.** We went from a model that thought X-rays were jellyfish to one with 99% pneumonia detection in under 5 minutes of training.

3. **Metrics matter in medicine.** Accuracy alone is misleading. In a screening context, recall (catching every sick patient) is more important than precision (avoiding false alarms). Our model has excellent recall (99%) but moderate precision (82%), which is the right trade-off for screening.

4. **More training isn't always better.** The stretch challenge showed that unfreezing more layers and training longer actually hurt test performance due to overfitting. With limited data, simpler models often generalize better.

5. **Data quality and quantity drive results.** The dataset has 3× more pneumonia than normal images. A production system would need more balanced data, data augmentation, and external validation on X-rays from different hospitals before real-world deployment.

---

*Lab 4 Explainer — CCI Session 4 | King Hussein Cancer Center — AI & Data Intelligence Department*
