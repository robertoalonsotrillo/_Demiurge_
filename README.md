# DEMIURGE
![CI](https://github.com/buganart/descriptor-transformer/workflows/CI/badge.svg?branch=main)
[![codecov](https://codecov.io/gh/buganart/descriptor-transformer/branch/main/graph/badge.svg)](https://codecov.io/gh/buganart/descriptor-transformer)

## Development

### Installation

    pip install -e .


### Install package and development tools

Install `python >= 3.6` and `pytorch` with GPU support if desired.

    pip install -r requirements.txt


<!-- Run the tests

    pytest -->

<!-- 
### Option 2: Using nix and direnv

1. Install the [nix](https://nixos.org/download.html) package manager
and [direnv](https://direnv.net/).
2. [Hook](https://direnv.net/docs/hook.html) `direnv` into your shell.
3. Type `direnv allow` from within the checkout of this repository. -->

## INTRODUCTION
*Demiurge* is a tripartite neural network architecture devised to generate and sequence musical sounds in the waveform domain (Donahue et al. 2019). The architecture combines a synthesis engine based on a **UNAGAN** + **MelGAN** model with a custom **NEURAL SEQUENCER**. The diagram below explains the relation between the different elements.

![Demiurge_1](https://user-images.githubusercontent.com/68105693/115943995-d0a6f200-a4e5-11eb-8a22-66212b2c315f.png)
The audio generation and sequencing neural-network-based processes work as follows:

1. Modified versions of **[melGAN](https://github.com/buganart/melgan-neurips)** (a vocoder that is a convolutional non-autoregressive feed-forward adversarial network ) and **[UNAGAN](https://github.com/buganart/unagan)** (an auto-regressive unconditional sound generating boundary-equilibrium GAN) will first process audio files `.wav` from an original database `RECORDED AUDIO DB` to produce GAN-generated `.wav` sound files, which are compiled into a new database `RAW GENERATED AUDIO DB`. 

2. The **descriptor model** in the **[NEURAL SEQUENCER](https://github.com/buganart/descriptor-transformer)** extracts a series of Los Mel Frequency Cepstral Coeﬃcients `MFCC` strings `.json` from the audio files in the `PREDICTOR DB` while the **predictor**, a time-series prediction model, generates projected descriptor sequences based on that data. 

3. As the predicted descriptors are just statistical values and need to be converted back to audio, a **query engine** matches the predicted descriptors based on the   `PREDICTOR DB` with those extracted from the `RAW GENERATED AUDIO DB`. The model then replaces the matched with the predicted descriptors using the audio reference from the `RAW GENERATED AUDIO DB`, merging and combining the resultant sound sequences into an output `.wav` audio file.

Please bear in mind that our model uses **[WandB](https://wandb.ai/)** to track and monitor training.

## SYNTHESIS ENGINE (melGAN + unaGAN)

The chart below explains the GAN-based sound synthesis process. Please bear in mind that for ideal results the **melGAN** and **UNAGAN** audio databases should be the same. Cross-feeding between different databases generates unpredictable (although sometimes musically interesting) results. Please record the `wandb_run_ids` for the final sound generation process. 

![melgan/unagan workflow](https://github.com/robertoalonsotrillo/descriptor-transformer/blob/main/_static/img/Demiurge.png)

### 1. melGAN

**[melGAN](https://github.com/buganart/melgan-neurips)**  (Kumar et al. 2019) is a fully convolutional non-autoregressive feed-forward adversarial network that uses mel-spectrograms as a lower-resolution audio representation model that can be both efficiently computed from and inverted back to raw audio format. An average melGAN run on [Google Colab](https://colab.research.google.com/) using a single V100 GPU may need a week to produce satisfactory results. The results obtained using a multi-GPU approach with parallel data vary. To train the model please use the following [notebook](https://colab.research.google.com/drive/1xUrh2pNUBTMO4s4YPsxAbUdTdlHjTeVU?usp=sharing).

<img width="957" alt="melgan" src="https://user-images.githubusercontent.com/68105693/115818429-53b94100-a42f-11eb-9cb5-1c6c20ba5243.png">

### 2. UNAGAN

**[UNAGAN](https://github.com/buganart/unagan)** (Liu et al. 2019) is an auto-regressive unconditional sound generating boundary-equilibrium GAN (Berthelot et al. 2017) that takes variable-length sequences of noise vectors to produce variable-length mel-spectrograms. A first UNAGAN model was eventually revised by Liu et al. at [Academia Sinica](https://musicai.citi.sinica.edu.tw) to improve the resultant audio quality by introducing in the generator a hierarchical architecture  model and circle regularization to avoid mode collapse. The model produces satisfactory results after 2 days of training on a single V100 GPU. The results obtained using a multi-GPU approach with parallel data vary. To train the model please use the following [notebook] (https://colab.research.google.com/drive/1JEXcGs-zVoAi84e79OO-7_G9qD3ptNwF?usp=sharing). 

### 3. Generator

After training **melGAN** and **UNAGAN**, you will have to use the **[SOUND GENERATOR](https://github.com/buganart/descriptor-transformer/blob/main/predict_notebook/Unagan_generate.ipynb)** to ouput `.wav` audio files. Please set the `melgan_run_id` and `unagan_run_id` created in the previous training steps. The output `.wav` files will be saved to the `output_dir` specified in the notebook. To train the model please use the following [notebook](https://colab.research.google.com/github/buganart/unagan/blob/master/Unagan_generate.ipynb)

## NEURAL SEQUENCER

### 1. Descriptor Model

As outlined above the **descriptor model** plays a crucial role in the the prediction workflow. You may use pretrained descriptor data by selecting a `wandb_run_id` from the **[DESCRIPTOR MODEL](https://github.com/robertoalonsotrillo/descriptor-transformer/blob/main/predict_notebook/descriptor_model_predict.ipynb)** or train your own model using this [notebook](https://colab.research.google.com/github/buganart/descriptor-transformer/blob/main/predict_notebook/descriptor_model_predict.ipynb), following the instructions indicated there. Four different time-series predictors were implemented as training options: 

- **LSTM** (Hochreiter et al. 1997)
- **LSTMEncoderDecoderModel** (Cho et al. 2014)
- **TransformerEncoderOnlyModel**
-  **TransformerModel** (Vaswani et al. 2017)

Both the "LSTM" and "TransformerEncoderOnlyModel" are one step prediction models, while "LSTMEncoderDecoderModel" and "TransformerModel" can predict descriptor sequences with specified sequence length. Once you train the model, record the `wandb_run_id` and paste it on the [prediction notebook](https://github.com/buganart/descriptor-transformer/blob/main/predict_notebook/descriptor_model_predict.ipynb). Then, provide paths to the RAW generated audio DB and Prediction DB, and run the notebook. The notebook will generate new descriptors from the descriptor model and convert them back into audio.

#### Training (script alternative)

To train the descriptor model, run

    python desc/train_function.py --selected_model <1 of 4 models above> --audio_db_dir <path to database> --window_size <input sequence length> --forecast_size <output sequence length>

The audio database should only contain files in `.wav` format. 


### 2. Prediction Model

The prediction workflow can be described in the diagram below:
![Demiurge1](https://user-images.githubusercontent.com/68105693/115944060-3f844b00-a4e6-11eb-8ba8-9c6c7f63f61c.png)

1. The prediction database will be processed into **descriptor input (descriptor database II)** for the descriptor model, and the descriptor model will *predict the subsequent descriptors* based on the input.
2. The audio database will be processed into **descriptor database I** that each descriptor will have *ID reference* back to the audio segment. 
3. The **query function** will replace the predicted new descriptors from the descriptor model with the closest match in the **descriptor database I** based on the distance function.
4. The audio segments referenced by the replaced descriptors from the query function will be combined and merged into a new audio file.

The [prediction notebook](https://github.com/buganart/descriptor-transformer/blob/main/predict_notebook/descriptor_model_predict.ipynb) for the descriptor model is located in [predict_notebook/](https://github.com/buganart/descriptor-transformer/tree/main/predict_notebook).

Follow the instruction in the [prediction notebook](https://github.com/buganart/descriptor-transformer/blob/main/predict_notebook/descriptor_model_predict.ipynb) to generate new descriptor and convert them back to audio.

