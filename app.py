import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import librosa
# import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import io
import tempfile

def animate_circular_audio_visualizer(audio_data):
    # Save the audio data to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
        tmp.write(audio_data.getvalue())
        tmp_filename = tmp.name

    y, sr = librosa.load(tmp_filename, sr=None, mono=False)
    if y.ndim <= 1:
        raise ValueError("Audio file does not have two channels")

    y = y / np.max(np.abs(y))
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    num_lines = 100
    theta = np.linspace(0, 2 * np.pi, num_lines, endpoint=False)
    lines = [ax.plot([t, t], [0, 0.5], color="lightblue")[0] for t in theta]
    
    segment_length = 1024

    def update(frame):
        start = frame * segment_length
        if start + segment_length > len(y[0]):
            return lines
        left_channel = y[0, start:start + segment_length]
        right_channel = y[1, start:start + segment_length]
        radii = np.abs(left_channel + right_channel)
        radii = np.interp(radii, (radii.min(), radii.max()), (0.1, 1))
        for line, radius in zip(lines, np.tile(radii, num_lines // len(radii) + 1)[:num_lines]):
            line.set_ydata([0, radius])
            color_intensity = np.interp(radius, [0.1, 1], [0.8, 0.3])
            line.set_color((color_intensity, color_intensity, 1))
        return lines

    ani = FuncAnimation(fig, update, frames=range(len(y[0]) // segment_length), blit=True, interval=30)
    with open("myvideo.html","w") as f:
        print(ani.to_html5_video(), file=f)
    
    HtmlFile = open("myvideo.html", "r")
    #HtmlFile="myvideo.html"
    source_code = HtmlFile.read() 
    components.html(source_code, height = 900,width=900)
    st.audio(tmp_filename)
    return y, sr
def main():
    st.title('Audio Visualizer')
    audio_file = st.file_uploader("Upload Audio", type=['mp3', 'wav', 'ogg'])

    if audio_file is not None:
        audio_data = io.BytesIO(audio_file.read())  # Read the audio file into a BytesIO buffer
        with st.spinner('Generating visualization...'):
            y, sr = animate_circular_audio_visualizer(audio_data)

if __name__ == "__main__":
    main()
