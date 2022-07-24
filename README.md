# Mini-DeepText: An Open-source Neural Text Experimental Toolkit (Current version: v2.0)

## Introduction

Mini-DeepText aims to quickly implement neural models for text classification, semantic matching, text generation and machine translation tasks. A notable feature is that Mini-DeepText currently provides a variety of NLP model training configurations, including Transformer models. You can train a variety of task models with custom hyperparameters by simply providing data and modifying the training profile.

## Usage

#### Go to the folder of Mini-DeepText-2.0 and do the following:
    python train.py train.json

## Input Data Format

    JSON example:

    {
        "doc_label": ["Computer--MachineLearning--DeepLearning", "Neuro--ComputationalNeuro"],
        "doc_token": ["I", "love", "deep", "learning"],
        "doc_keyword": ["deep learning"],
        "doc_topic": ["AI", "Machine learning"]
    }

    "doc_keyword" and "doc_topic" are optional.
    
## Configuration File

    "data_info": {
        "data_filepath": "Datasets/nmt_data_word.json",
        "split_ratio": [
            0.7,
            0.15,
            0.15
        ],
        "max_seq_length": 30,
        "cutoff": 0,
        "seed": 1337
    },
    "train_info": {
        "num_epochs": 10,
        "batch_size": 64,
        "optimizer": "adam",
        "learning_rate": 1e-4,
        "early_stopping_criteria": 3
    },
    "save_info": {
        "save_folder": "ModelFolder",
        "vectorizer_file": "ModelFolder/trans_word_vectorizer.json",
        "model_state_file": "ModelFolder/trans_word_model.pth"
    },
    "task": "translation",
    "model_name": "Transformer",
    "model": {
        "TextCLRModel": {
            "embedding_size": 128,
            "rnn_hidden_size": 64
        },
        "TextDSMModel": {
            "embedding_size": 128,
            "rnn_hidden_size": 64
        },
        "TextSLBModel": {
            "embedding_size": 128,
            "rnn_hidden_size": 64
        },
        "TextNMTModel": {
            "source_embedding_size": 128,
            "target_embedding_size": 128,
            "encoding_size": 64
        },
        "Transformer": {
            "source_embed_dim": 512,
            "target_embed_dim": 512,
            "encoder_n_heads": 8,
            "decoder_n_heads": 8,
            "encoder_hid_dim": 2048,
            "decoder_hid_dim": 2048,
            "encoder_n_layers": 6,
            "decoder_n_layers": 6
        }
    }

## Support tasks

* Multi-class text classifcation
* Semantic matching
* Text generation
* Machine Translation

## Support NLP model
* RNN text classification model
* RNN semantic matching model
* RNN text generation model
* RNN machine translation model
* Transformer model

## Requirement
* Python 3
* PyTorch 0.4+
* Numpy 1.14.3+
