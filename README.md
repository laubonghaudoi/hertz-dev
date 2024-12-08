# hertz-dev

Hertz-dev is an open-source, first-of-its-kind base model for full-duplex conversational audio.

See our blog post for more details: https://si.inc/hertz-dev/

## Setup

Inference is known to work on Python 3.10 and CUDA 12.1. Other versions have not been tested as thoroughly. If you want to use CUDA 12.1, you'll need to install torch with `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121` before running `pip install -r requirements.txt`.

On Ubuntu you may need to install libportaudio: `sudo apt-get install libportaudio2`

All three scripts will automatically download the models to the `./ckpt` directory, and checkpoints are also accessible at https://ckpt.si.inc/hertz-dev/index.txt

## Data Preparation

1. Create data directories:
```bash
mkdir -p data/train data/val
```

2. Prepare your audio data:
   - Place training audio files in `data/train/`
   - Place validation audio files in `data/val/`
   - Supported formats: WAV and MP3
   - Audio will be automatically:
     - Converted to mono if stereo
     - Resampled to 16kHz
     - Cropped to 6-second segments during training

## Training

The training process is split into two phases:

### 1. Train the Codec

First, train the audio codec which learns to compress audio into latent representations:

```bash
python train_codec.py
```

Key configurations in `config/codec_config.py`:
- Batch size, learning rate, and training steps
- Model architecture parameters
- Loss weights for reconstruction and KL divergence
- Hardware settings (device, precision)

Checkpoints will be saved to `checkpoints/codec/`

### 2. Train the Transformer

After the codec is trained, train the transformer model which learns to predict audio tokens:

```bash
python train_transformer.py
```

Key configurations in `config/transformer_config.py`:
- Model architecture (layers, heads, dimensions)
- Training parameters (batch size, learning rate)
- Sequence length and vocabulary size
- Hardware settings

Checkpoints will be saved to `checkpoints/`

Training progress for both models can be monitored through Weights & Biases.

## Usage

We recommend starting by using `inference.ipynb` to generate one- or two-channel completions from a prompt.

Then, you can use `inference_client.py` and `inference_server.py` to talk to the model live through your microphone.
These are currently experimental, and have primarily been tested with Ubuntu on the server and MacOS on the client.

## Training Tips

1. Start with a small dataset to verify the training pipeline
2. Monitor the reconstruction loss for the codec training
3. The transformer training may take longer - use gradient accumulation for larger batch sizes
4. Use mixed precision training (enabled by default) for faster training
5. Adjust temperatures during inference to control output diversity

