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

## Description
*Demiurge* is a tri-modal Generative Adversarial Network (Goodfellow et al., 2014) architecture devised to generate and sequence musical sounds in the waveform domain (Donahue et al., 2019). The architecture combines UnaGAN plus MelGAN models as a sound generators with a custom GAN sequencer. The diagram below explains the relation between the different elements.
![project concept](https://github.com/robertoalonsotrillo/descriptor-transformer/blob/main/_static/img/Dataflow.png)

The audio generation and sequencing GAN-based processes work as follows:

1. Modified versions of **[MELGAN](https://github.com/buganart/melgan-neurips)** (a vocoder that is a convolutional non-autoregressive feed-forward adversarial network ) and **[UNAGAN](https://github.com/buganart/unagan)** (an auto-regressive unconditional sound generating boundary-equilibrium GAN) will first process audio files (.wav) from an original database `RECORDED AUDIO DB` to produce GAN-generated sound files (.wavs), compiled into a new database `RAW GENERATED AUDIO DB`. 

2. The **[DESCRIPTOR MODEL](https://github.com/buganart/descriptor-transformer)** extracts a series of MFCC descriptor strings (.json) from the audio files in the `PREDICTOR DB` and the sequencer, which is the time series prediction model in the current repository, generates projected descriptor sequences based on that data. 

3. As the predicted descriptors are just statistical values and need to be converted back to audio, a query engine matches the predicted descriptors based on the   `PREDICTOR DB` with those extracted from the `RAW GENERATED AUDIO DB`. The model then replaces the audio reference taken from the the `RAW GENERATED AUDIO DB` of the matched extracted descriptors with that of the predicted descriptors, merging and combining it into an output prediction audio file (.wav).


## Training (SEQUENCER GAN)

From the descripton above, descriptor model(SEQUENCER GAN) is necessary for the prediction workflow. User can use one of the pretrained descriptor model with the wandb run id in the [prediction notebook](https://github.com/buganart/descriptor-transformer/blob/main/predict_notebook/descriptor_model_predict.ipynb), or train their own model with the instruction in the training section below.

For the descriptor model, there are 4 models to choose from: "LSTM", "LSTMEncoderDecoderModel", "TransformerEncoderOnlyModel", or "TransformerModel".
The "LSTM" and "TransformerEncoderOnlyModel" are one step prediction model, while "LSTMEncoderDecoderModel" and "TransformerModel" can predict descriptor sequence with specified sequence length.

After training the model, record the wandb run id and paste it in the [prediction notebook](https://github.com/buganart/descriptor-transformer/blob/main/predict_notebook/descriptor_model_predict.ipynb). Then, provide paths to the RAW generated audio DB and Prediction DB, and run the notebook. The notebook will generate new descriptors from the descriptor model and convert them back into audio.

### Training (notebook)

The [training notebook](https://github.com/buganart/descriptor-transformer/blob/main/train_notebook/descriptor_model_train.ipynb) for the descriptor model is located in the folder [train_notebook/](https://github.com/buganart/descriptor-transformer/tree/main/train_notebook).

Follow the instruction in the [training notebook](https://github.com/buganart/descriptor-transformer/blob/main/train_notebook/descriptor_model_train.ipynb) to train the descriptor model.

### Training (script)

To train the descriptor model, run

    python desc/train_function.py --selected_model <1 of 4 models above> --audio_db_dir <path to database> --window_size <input sequence length> --forecast_size <output sequence length>

The audio database shoulf be audio file in ".wav"


## Prediction

The prediction workflow can be described in the diagram below:

![descriptor workflow](https://github.com/buganart/descriptor-transformer/blob/main/_static/img/descriptor_model_predict_workflow.png)

1. The prediction database will be processed into **descriptor input (descriptor database II)** for the descriptor model, and the descriptor model will *predict the subsequent descriptors* based on the input.
2. The audio database will be processed into **descriptor database I** that each descriptor will have *ID reference* back to the audio segment. 
3. The **query function** will replace the predicted new descriptors from the descriptor model with the closest match in the **descriptor database I** based on the distance function.
4. The audio segments referenced by the replaced descriptors from the query function will be combined and merged into a new audio file.

The [prediction notebook](https://github.com/buganart/descriptor-transformer/blob/main/predict_notebook/descriptor_model_predict.ipynb) for the descriptor model is located in [predict_notebook/](https://github.com/buganart/descriptor-transformer/tree/main/predict_notebook).

Follow the instruction in the [prediction notebook](https://github.com/buganart/descriptor-transformer/blob/main/predict_notebook/descriptor_model_predict.ipynb) to generate new descriptor and convert them back to audio.

## Optional: melgan and unagan

The melgan and unagan is used to generate a lot more audio files that are similar to the files in the RECORDED AUDIO DB. This is optional if the RECORDED AUDIO DB is already large enough for the descriptor matching process in the query function.


![melgan/unagan workflow](https://github.com/buganart/descriptor-transformer/blob/main/_static/img/sound_generation_process.png)

For the [melgan](https://github.com/buganart/descriptor-transformer/blob/main/train_notebook/melgan.ipynb)/[unagan](https://github.com/buganart/descriptor-transformer/blob/main/train_notebook/unagan.ipynb) training, please use notebooks in the folder [train_notebook/](https://github.com/buganart/descriptor-transformer/tree/main/train_notebook).
The audio database for the melgan and unagan should be the same, and please record wandb run id of the run for sound generation.

After the melgan and unagan are trained, go to [unagan generate notebook](https://github.com/buganart/descriptor-transformer/blob/main/predict_notebook/Unagan_generate.ipynb) and set the melgan_run_id and unagan_run_id. The output wav files will be saved to the output_dir specified in the notebook.
