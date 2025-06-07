# Denoising Autoencoder on Signal wave

## Overview
This project implements a **Denoising Autoencoder** using TensorFlow to remove noise from synthetic signal wave data. The autoencoder is designed to learn how to reconstruct clean signals from noisy inputs, leveraging a convolutional neural network (CNN) architecture with 1D convolutions, max-pooling, and upsampling layers.

## Dataset
The dataset is synthetically generated using the `generate_wave_data` function, which creates sequences of signal waves by combining two cosine functions with random frequencies and amplitudes, then adding Gaussian noise. Key parameters:
- **Number of sequences**: 1000
- **Sequence length**: 256
- **Noise level**: 0.1 (standard deviation of Gaussian noise)

The data is split into 80% training and 20% testing sets using `train_test_split` from scikit-learn.

## Model Architecture
The denoising autoencoder consists of an encoder and a decoder:
- **Encoder**:
  - Input shape: `(256, 1)` (sequence length, channels)
  - Conv1D (16 filters, kernel size 3, ReLU, causal padding, dilation rate 1)
  - MaxPooling1D (pool size 2)
  - Conv1D (32 filters, kernel size 3, ReLU, causal padding, dilation rate 2)
  - MaxPooling1D (pool size 2)
  - Conv1D (64 filters, kernel size 3, ReLU, causal padding, dilation rate 4)
- **Decoder**:
  - UpSampling1D (size 2)
  - Conv1D (32 filters, kernel size 3, ReLU, same padding)
  - UpSampling1D (size 2)
  - Conv1D (16 filters, kernel size 3, ReLU, same padding)
  - Conv1D (1 filter, kernel size 3, linear activation, same padding)

The model is compiled with:
- **Optimizer**: Adam (learning rate = 0.001)
- **Loss function**: Mean Squared Error (MSE)

## Training
The autoencoder is trained for 30 epochs with a batch size of 32. The training and validation loss (MSE) are plotted to evaluate convergence.

## Evaluation
The model is evaluated on the test set using:
- **Mean Squared Error (MSE)**: Measures the average squared difference between the noisy input and denoised output.
- **Mean Absolute Error (MAE)**: Measures the average absolute difference between the noisy input and denoised output.

Example results (from the provided output):
- MSE: ~104.58
- MAE: ~5.12

Visualizations of the first 10 test samples compare the noisy input signals with their denoised outputs.

## Files
- **Denoising_Autoencoder_on_Signal_wave_data.py**: Python script containing the complete implementation, including data generation, model definition, training, and evaluation.
- **Denoising_Autoencoder_on_Signal_wave_data.ipynb**: Jupyter Notebook version of the script, originally run on Google Colab with GPU support.

## Requirements
To run the code, install the following Python packages:
```bash
pip install tensorflow numpy matplotlib scikit-learn
```

## Usage
1. Clone or download the repository.
2. Ensure the required packages are installed.
3. Run the Python script or Jupyter Notebook:
   ```bash
   python Denoising_Autoencoder_on_Signal_wave_data.py
   ```
   or open `Denoising_Autoencoder_on_Signal_wave_data.ipynb` in Jupyter/Colab.
4. The script will:
   - Generate synthetic signal wave data.
   - Train the autoencoder.
   - Plot training/validation loss and sample input/output comparisons.
   - Output MSE and MAE metrics.

## Notes
- The model assumes access to a GPU for faster training, but it can run on a CPU.
- The synthetic data generation can be modified by adjusting `num_sequences`, `sequence_length`, or `noise_level` in the script.
- The provided MSE and MAE values are specific to the run in the notebook; results may vary slightly due to randomness in data generation and training.

## Future Improvements
- Experiment with different noise levels or signal types (e.g., sine waves, sawtooth waves).
- Add regularization (e.g., dropout) to prevent overfitting.
- Explore deeper architectures or alternative models like LSTM-based autoencoders.
- Save and load the trained model for reuse.

## License
This project is licensed under the MIT License.
