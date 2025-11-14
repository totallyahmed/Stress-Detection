import gradio as gr
import numpy as np
import librosa
import matplotlib.pyplot as plt
import os
from keras.models import load_model
import librosa.display

# ------------------------------
# Load Keras model
# ------------------------------
MODEL_PATH = "model/stress_detection_keras.h5"
model = load_model(MODEL_PATH)

# ------------------------------
# Global audio settings
# ------------------------------
SAMPLE_RATE = 16000
CLIP_SEC = 3
CLIP_LEN = SAMPLE_RATE * CLIP_SEC
N_MFCC = 40
MAX_LEN = 130


# ------------------------------
# Preprocessing utilities
# ------------------------------
def load_audio_array(audio):
    """
    Takes gradio audio input and returns standardized waveform.
    """
    y, sr = librosa.load(audio, sr=SAMPLE_RATE)

    if len(y) < CLIP_LEN:
        y = np.pad(y, (0, CLIP_LEN - len(y)))
    else:
        y = y[:CLIP_LEN]

    return y


def extract_mfcc(y):
    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC)

    if mfcc.shape[1] < MAX_LEN:
        pad_len = MAX_LEN - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_len)))
    else:
        mfcc = mfcc[:, :MAX_LEN]

    return mfcc[..., np.newaxis]


# ------------------------------
# Visualization helpers
# ------------------------------
def plot_waveform(y):
    plt.figure(figsize=(10, 3))
    librosa.display.waveshow(y, sr=SAMPLE_RATE, color="#1f77b4")
    plt.title("Waveform")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")

    plot_path = "figures/waveform.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    return plot_path


def plot_mfcc(mfcc):
    plt.figure(figsize=(6, 4))
    librosa.display.specshow(mfcc[:, :, 0], sr=SAMPLE_RATE, x_axis='time', cmap='inferno')
    plt.title("MFCC")
    plt.colorbar(format="%+2.f dB")

    plot_path = "figures/mfcc.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    return plot_path


# ------------------------------
# Main inference function
# ------------------------------
def predict_stress(audio):
    if audio is None:
        return "Please upload/record audio.", None, None

    # audio['name'] if upload, array if record
    if isinstance(audio, dict):
        file_path = audio['name']
    else:
        file_path = audio

    y = load_audio_array(file_path)
    mfcc = extract_mfcc(y)

    pred = model.predict(np.expand_dims(mfcc, axis=0))[0][0]

    label = "Stressed" if pred >= 0.5 else "Non-Stressed"
    confidence = pred if pred >= 0.5 else 1 - pred
    confidence = float(confidence)

    # Create plots
    waveform_img = plot_waveform(y)
    mfcc_img = plot_mfcc(mfcc)

    prediction_text = f"**Prediction:** {label}\n**Confidence:** {confidence*100:.2f}%"

    return prediction_text, waveform_img, mfcc_img


# ------------------------------
# Build Gradio Interface
# ------------------------------
css = """
#prediction-text {
    font-size: 22px !important;
    font-weight: bold;
}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=css) as app:
    gr.Markdown("# 🎤 Speech-Based Stress Detection")
    gr.Markdown("Upload or record 3 seconds of speech to detect stress levels.")

    with gr.Row():
        audio_input = gr.Audio(
            sources=["microphone", "upload"], 
            type="filepath", 
            label="Upload or Record Audio"
        )

    with gr.Row():
        prediction = gr.Markdown("", elem_id="prediction-text")

    with gr.Row():
        waveform_output = gr.Image(label="Waveform")
        mfcc_output = gr.Image(label="MFCC")

    btn = gr.Button("Analyze Stress Level")
    btn.click(
        fn=predict_stress, 
        inputs=audio_input, 
        outputs=[prediction, waveform_output, mfcc_output]
    )

# ------------------------------
# Launch
# ------------------------------
if __name__ == "__main__":
    app.launch()
