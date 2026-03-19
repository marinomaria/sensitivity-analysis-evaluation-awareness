# Project description

This project is the setup for our experiments on the thesis "Sensitivity Analysis of Evaluation Awareness in Large Languaje Models". We have datasets: `datasets/contrastive_dataset.json` is aimed to train a linear probe mimicking `../evaluation-awareness-probing` and `datasets/sensitivity_dataset.json` is a set of prompts we want to classify with the probe.

This project does the following things:

1. Trains a linear probe mimicking `../evaluation-awareness-probing` and using the contrastive dataset
2. Runs the sensitivity dataset through the model
3. In the `output` folder it stores the resulting dataset. It consists of the sensitivity dataset plus, for each prompt
    1. the text output of the model
    2. the float representing the projection of the prompt on the probe
    3. the veredict given by the probe


# Linear probes training and usage

For all linear probe related features mimic the `../evaluation-awareness-probing` repository.

# Models

This project uses the Llama family. It should be easy to switch from Llama-3.2-1B-Instruct (test-mode) to Llama-3.3-70B-Instruct. In the future we will want to add more open-weights models from HuggingFace like Qwen and DeepSeek, so the design must allow that.