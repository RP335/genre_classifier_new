import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display


def plot_mel_spectrogram(audio_file, ax, title):
    y, sr = librosa.load(audio_file, sr=None)

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_DB = librosa.amplitude_to_db(S, ref=np.max)

    img = librosa.display.specshow(S_DB, sr=sr, x_axis='time', y_axis='mel', ax=ax)
    ax.set(title=title)
    ax.label_outer()
    return img


audio_files = [
    ('path_to_audio_file_blues.wav', 'Blues'),
    ('path_to_audio_file_classical.wav', 'Classical'),
    ('path_to_audio_file_disco.wav', 'Disco'),
    ('path_to_audio_file_reggae.wav', 'Reggae')
]

fig, axs = plt.subplots(2, 2, figsize=(10, 8))

for (audio_file, title), ax in zip(audio_files, axs.flatten()):
    plot_mel_spectrogram(audio_file, ax, title)

# Display the plots
plt.tight_layout()
plt.show()
