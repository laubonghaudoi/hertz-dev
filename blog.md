# Introducing hertz-dev - Standard Intelligence

For the last few months, the team at Standard Intelligence has been doing research in _cross-modality learning_. We're excited to announce that we're [open-sourcing](https://ckpt.si.inc/hertz-dev/index.txt) an early product of this research, an **8.5B, full-duplex, audio-only base model: hertz-dev**.

Audio modality is imperative to creating interactive agents that feel natural. Currently the two methods of utilizing audio with generative AI are either _diffusion based methods_ or _autoregressive methods_. Though diffusion based audio models prove to be good at music generation and small samples, truly interactive audio generation needs to be autoregressive.

The largest problems in this field are **1)** Getting audio generation that sounds human (ie. non-synthetic as well as handling interruptions well) and **2)** Handling realtime generation with two live channels that are both producing information, like regular human dialogue.

Our model is at the frontier of both of these, natively fitting to the two-speaker format with faster-than-human reaction times and full ability to parse and generate overlapping two-speaker audio. We do this by operating in latent space as well as using quantized phonetic bits, allowing a **80ms theoretical average latency** with only a single sampled latent at each timestep. Currently, we benchmark at **120ms real-world latency** on a single RTX 4090—2x lower than the previous state of the art.

## Overview

![hertz-codec architecture diagram](https://si.inc/static/hertz-codec.png)

Figure 1: `hertz-codec` architecture diagram for our VAE. The input is 6s 16kHz mono audio and the output is a 32-dim latent.

![hertz-ar architecture diagrams](https://si.inc/static/hertz-ar.png)

Figure 2: `hertz-ar` architecture diagram for the autoregressive section of our model. (2a) is mono-channel autoregressive latent prediction and (2b) is duplex autoregressive latent prediction.

`hertz-dev` is made out of two parts—the `hertz-codec` which produces audio latents and the `hertz-ar` which predicts future latents conditioned on past latents. The audio latents are an extremely rich prior that could be used for many downstream tasks.

- **hertz-codec**: a convolutional audio VAE which takes mono, 16kHz speech and encodes a 8Hz latent representation with a KL-regularized 1kbps bitrate. We utilize causal convolutions (functionally adding padding to the left of the sequence) to enable streaming inference.

  The codec outputs gaussian parameters (means and variances) that are sampled into just a single 32-dim latent per 125ms frame. Hertz-codec outperforms Soundstream and Encodec at 6kbps and is on par with DAC at 8kbps in subjective evaluations, while having lower tokens per second than any popular tokenizer, critical for language modeling. Hertz-codec has 5 million encoder parameters and 95 million decoder parameters.

  - [inference_apatosaurus_95000.pt](https://ckpt.si.inc/hertz-dev/inference_apatosaurus_95000.pt) — `hertz-codec` weights trained on a mixed reconstruction, adversarial, and KL-regularized loss.
  - [inference_volcano_3.pt](https://ckpt.si.inc/hertz-dev/inference_volcano_3.pt) — `hertz-codec quantizer`, a learned projection distilling the most phonetically relevant 15-bits of each latent.

- **hertz-ar**: a 40-layer 8.4 billion parameter decoder-only transformer with a context of 2048 input tokens (~4.5 mins). The output is a latent that can be passed into hertz-codec. The first 32 layers receive as input the latent history and predict a 15 bit quantized projection of the next latent audio token. We call this the `hertz-lm` as it can either be trained independently or initialized from language model weights.

  The last 8 layers then utilize the latent history and the 15 bit quantized latent to predict future latent audio tokens.

  Duplex audio is handled as a post-training task with two projection heads concatenated together, then separated into two quantized projection pipelines conditioned on their respective residuals.

  - [inference_caraway_112000.pt](https://ckpt.si.inc/hertz-dev/inference_caraway_112000.pt) — `hertz-lm` weights initialized from a language model trained on 2T tokens.
  - [inference_syrup_110000.pt](https://ckpt.si.inc/hertz-dev/inference_syrup_110000.pt) — `hertz-lm` weights initialized randomly and fully trained on audio latents.
  - [inference_whip_72000.pt](https://ckpt.si.inc/hertz-dev/inference_whip_72000.pt) — `hertz-ar` weights for the last 8 layers
  - [inference_care_50000.pt](https://ckpt.si.inc/hertz-dev/inference_care_50000.pt) & [inference_scion_54000.pt](https://ckpt.si.inc/hertz-dev/inference_scion_54000.pt) — Duplex checkpoints for `hertz-ar`

Hertz-dev is the first publicly released base model for conversational audio. Base models accurately predict the distribution of the data that they were trained on, as opposed to models that have had substantial RL tuning done to collapse their generation distributions. This makes these models the best starting point for downstream fine-tuning in a large number of different tasks. We're currently training a larger, more advanced version of Hertz, which will use a scaled base model recipe and RL tuning to substantially improve the raw capabilities and final coherence of the model. Hertz-dev is a glimpse at the future of real-time voice interaction, and is the easiest conversational audio model in the world for researchers to fine-tune and build on top of.

## Sample Generations

To demonstrate the audio modeling capabilities of hertz-dev, we sample both one-channel and two-channel generations as well as a live conversation between the model and a human.

### One-channel

Your browser does not support the audio element.

Your browser does not support the audio element.

Your browser does not support the audio element.

Your browser does not support the audio element.

### Two-channel

Your browser does not support the audio element.

Your browser does not support the audio element.

Your browser does not support the audio element.

### Interactive

Your browser does not support the audio element.

9 seconds of prompt included.

## Training Choices

- Causal ConvNets were used in hertz-codec for parallel decoding + more granular control over latent generation.
- 15bit quantized latents were initially trained to contain phonetics which helps steer the model to create syntactically correct speech. Quantization was done via an MLP projection into a Finite Scalar Quantization layer.
- We chose to ablate two separate initialization strategies for `hertz-lm`, and found that our model recipe effectively learns linguistics both with and without text model initialization.

## Performance

During live inference, the model needs to run at 8 forward passes per second, doing constant autoregressive generation. It takes two separate channels as input, but in conversations only returns one. At each step, it receives the human’s audio and tokenizes it into a latent, combining this with the model’s last generated latent and feeding both into `hertz-ar`.

This allows the latency, measured as the average time between user utterance and model response, to be 62.5ms (the average time between any given utterance and the end of one token) + the time for forward pass + round-trip internet delay. By running on local 4090s, we usually see a real-world average latency of 120ms. This is 2x lower than any other audio model—which is necessary for a model that can interact with you in human-like ways instead of what feels like a delayed, choppy phone call.
