---
title: "Wav2Lip: Accurately Lip-syncing Videos In The Wild"
version: "1.0"
license: "For research/academic purposes only – non-commercial use"
author: "Wav2Lip Authors"
tags: [lip-sync, video generation, deep learning]
---

# Wav2Lip: Accurately Lip-syncing Videos In The Wild

[![PWC](https://img.shields.io/badge/PWC-Paper-blue)](https://paperswithcode.com/paper/a-lip-sync-expert-is-all-you-need-for-speech) 
[![Project Page](https://img.shields.io/badge/Project-Page-green)](http://cvit.iiit.ac.in/research/projects/cvit-projects/a-lip-sync-expert-is-all-you-need-for-speech-to-lip-generation-in-the-wild/)
[![Demo](https://img.shields.io/badge/Demo-Video-orange)](https://youtu.be/0fXaDCZNOJc)

> **Note**: For commercial requests, please contact us at radrabha.m@research.iiit.ac.in or prajwal.k@research.iiit.ac.in. We have an HD model ready that can be used commercially.

This code is part of the paper: **"A Lip Sync Expert Is All You Need for Speech to Lip Generation In the Wild"** published at ACM Multimedia 2020.

## Quick Links
- 📑 [Original Paper](https://arxiv.org/abs/2008.10010)
- 📰 [Project Page](http://cvit.iiit.ac.in/research/projects/cvit-projects/a-lip-sync-expert-is-all-you-need-for-speech-to-lip-generation-in-the-wild/)
- 🌀 [Demo Video](https://youtu.be/0fXaDCZNOJc)
- ⚡ [Interactive Demo](https://colab.research.google.com/drive/1tZpDWXz49W6wDcTprANRGLo2D_EbD5J8?usp=sharing)
- 📔 [Colab Notebook](https://colab.research.google.com/drive/1tZpDWXz49W6wDcTprANRGLo2D_EbD5J8?usp=sharing)

## Highlights 🌟

- Lip-sync videos to any target speech with high accuracy 💯
- Works for any identity, voice, and language
- Supports CGI faces and synthetic voices ✨
- Complete training code, inference code, and pretrained models available 🚀
- Quick-start with Google Colab Notebook
- New evaluation benchmarks and metrics released 🔥

## Prerequisites

- Python 3.6
- ffmpeg: 
  ```bash
  sudo apt-get install ffmpeg
  ```
- Required packages:
  ```bash
  pip install -r requirements.txt
  ```
- Face detection pre-trained model: Download to `face_detection/detection/sfd/s3fd.pth`

## Available Models

| Model | Description | Download |
|-------|-------------|----------|
| Wav2Lip | Highly accurate lip-sync | [Download](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/Eb3LEzbfuKlJiR600lQWRxgBIY27JZg0XdhEGKhXJO9ceQ?e=TBFBVW) |
| Wav2Lip + GAN | Better visual quality | [Download](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/EdjI7bZlgApMqsVoEUUXpLsBxqXbn5z8VTmoxp55YNDcIA?e=n9ljGW) |
| Expert Discriminator | Expert discriminator weights | [Download](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/EQRvmiZg-HRAjvI6zqN9eTEBP74KefynCwPWVmF57l-AYA?e=ZRPHKP) |
| Visual Quality Discriminator | Visual disc (GAN) weights | [Download](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/EQVqH88dTm1HjlK11eNba5gBbn15WMS0B0EZbDBttqrqkg?e=ic0ljo) |

## Inference

Lip-sync any video to any audio:

```bash
python inference.py --checkpoint_path <ckpt> --face <video.mp4> --audio <an-audio-source>
```

### Tips for Better Results

1. Adjust face bounding box with `--pads`:
   ```bash
   python inference.py --pads 0 20 0 0 ...
   ```

2. Disable smoothing for misaligned results:
   ```bash
   python inference.py --nosmooth ...
   ```

3. Experiment with `--resize_factor` for different video resolutions

## Training

### 1. Prepare LRS2 Dataset

```bash
python preprocess.py --data_root data_root/main --preprocessed_root lrs2_preprocessed/
```

### 2. Train Expert Discriminator

```bash
python color_syncnet_train.py --data_root lrs2_preprocessed/ --checkpoint_dir <checkpoint_folder>
```

### 3. Train Wav2Lip Model

Without GAN (faster):
```bash
python wav2lip_train.py --data_root lrs2_preprocessed/ --checkpoint_dir <checkpoint_folder> --syncnet_checkpoint_path <expert_disc_checkpoint>
```

With GAN (better quality):
```bash
python hq_wav2lip_train.py --data_root lrs2_preprocessed/ --checkpoint_dir <checkpoint_folder> --syncnet_checkpoint_path <expert_disc_checkpoint>
```

## License and Citation

This repository is for personal/research/non-commercial purposes only. For commercial usage, please contact us directly.

If you use this repository, please cite our paper:

```bibtex
@inproceedings{10.1145/3394171.3413532,
  author = {Prajwal, K R and Mukhopadhyay, Rudrabha and Namboodiri, Vinay P. and Jawahar, C.V.},
  title = {A Lip Sync Expert Is All You Need for Speech to Lip Generation In the Wild},
  year = {2020},
  isbn = {9781450379885},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3394171.3413532},
  doi = {10.1145/3394171.3413532},
  booktitle = {Proceedings of the 28th ACM International Conference on Multimedia},
  pages = {484–492},
  numpages = {9},
  keywords = {lip sync, talking face generation, video generation},
  location = {Seattle, WA, USA},
  series = {MM '20}
}
```

## Acknowledgements

- Code structure inspired by [TTS repository](https://github.com/mozilla/TTS)
- Face Detection code from [face_alignment](https://github.com/1adrianb/face-alignment)
- Thanks to [zabique](https://github.com/zabique) for the tutorial collab notebook
