# NER-finetuning-test

## Introduction
This repo contains a simple implementation of finetuning a pre-trained model for Named Entity Recognition on the English subset of the MultiNERD dataset.

Two models are trained: one trained on all the classes in MultiNERD (ner_all) and one only trained only on the following five (ner_five): PERSON(PER), ORGANIZATION(ORG), LOCATION(LOC), DISEASES(DIS), ANIMAL(ANIM).

The majority of the code in this repo is copied directly from  [Tirendaz Academy's "NER with HuggingFace](https://www.kaggle.com/code/tirendazacademy/ner-with-huggingface) tutorial.

## How to use

One need only run the ner_finetuning.ipynb notebook to get the results. The code depends on the HuggingFace to source both the dataset ([MultiNERD](https://huggingface.co/datasets/Babelscape/multinerd)) and the pre-trained model used ([distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased)), thus an account is required to run it, and an access token will be requested in cell #12.

## Training parameters

Both models were trained with the same following training parameters:

learning_rate=2e-5
per_device_train_batch_size=32
per_device_eval_batch_size=32
num_train_epochs=2
weight_decay=0.005

There was no search attempt for optimal training parameters due to time constraints and lack of resources. Better results are almost certainly achievable (such as by increasing epoch number.) 

## Results

Model | Precision | Recall | F1 | Accuracy
| :--- | ---: | :---: | :---: | :---:
ner_all  | 0.9089245549159392 | 0.9296938974458959 | 0.9191919191919193 | 0.9868482097831568
ner_five  | 0.9603748077012727 | 0.9646017699115044 | 0.9624836479162773 | 0.9951127440386139

As can be seen, reducing the scope of the model to only the above five classes improves better performance. This is unsurprising, considering the five remaining classes are the ones with the highest amount of examples in the dataset, as can be seen in [the dataset's corresponding paper](https://aclanthology.org/2022.findings-naacl.60.pdf).
