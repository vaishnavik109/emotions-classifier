# emotions-classifier
# Speech Emotion Recognition using Deep Learning

## Objective

To design and implement a deep learning model that can recognize human emotions from speech by analyzing audio features using a neural network.

---

## Methodology

1. **Data Loading**
   Load audio samples from a structured dataset where each file corresponds to a labeled emotion.

2. **Preprocessing**  
   - Convert audio to mono format
   - Normalize the audio signal
   - Trim silence from start and end (optional)

3. **Feature Extraction**  
   Use the `librosa` library to extract relevant audio features from each sample. These features represent patterns of frequency and amplitude that are useful for emotion classification.

4. **Feature Aggregation**  
   Combine multiple features into a single feature vector for each sample.

5. **Label Encoding and Dataset Splitting**  
   Encode emotion labels as one-hot vectors and split the dataset into training and testing sets using `train_test_split`.

6. **Model Training**  
   - Use a feed-forward deep neural network built with TensorFlow/Keras
   - Compile with `Adam` optimizer and `categorical_crossentropy` loss
   - Train with early stopping or fixed epochs

7. **Evaluation and Visualization**  
   - Evaluate model on the test set
   - Plot accuracy and loss curves
   - Display confusion matrix

---

##Feature Extraction

The following features are extracted using `librosa`:

- **MFCC (Mel Frequency Cepstral Coefficients):** Represents short-term power spectrum of sound.
- **Chroma Frequencies:** Represents 12 different pitch classes.
- **Spectral Contrast:** Measures the difference between peaks and valleys in the spectrum.
- **Tonnetz Features:** Captures tonal characteristics (pitch & harmony).

These are concatenated into a final feature vector per audio file.

---

## Model Architecture

```text
Input Layer (Number of features)
↓
Dense Layer (128 units, ReLU)
↓
Dropout (rate=0.3)
↓
Dense Layer (64 units, ReLU)
↓
Dropout (rate=0.3)
↓
Dense Layer (Number of classes, Softmax)
