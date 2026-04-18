# Audio Processing

import matplotlib.pyplot as plt
import soundfile as sf

# TODO: Read an audio file
audio_data, sample_rate = None
print(f"Sample Rate: {sample_rate}, Audio Data Shape: {audio_data.shape}")

# TODO: Reverse the signal using flipud function
reversed_signal = None

# TODO: Reverse the signal using flipud function
reversed_channels = None

# TODO: Save the new audio files on your disk
reversed_signal_file = "reversed_signal.wav"
reversed_channels_file = "reversed_channels.wav"

# TODO: Read the new audio files
reversed_signal_data, _ = sf.read(reversed_signal_file)

# TODO: plot three signal on one figure
plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)




