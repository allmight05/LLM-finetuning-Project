```
Fine-Tuning FLAN-T5 with LoRA & Quantization for Question Answering

This project demonstrates how to fine-tune a pre-trained FLAN-T5 model using the LoRA (Low-Rank Adaptation) technique and quantize it with bitsandbytes to reduce memory usage. The model is fine-tuned on the SQuAD v2 dataset to perform real-world question answering tasks.
Project Overview

    Quantization: Configure the model for 4-bit quantization using BitsAndBytes to reduce memory usage.
    Model Initialization: Load and initialize the FLAN-T5 model with the quantization configuration.
    Model Freezing & Gradient Checkpointing: Freeze base model parameters and enable gradient checkpointing to lower memory consumption during training.
    Pre-Fine-Tuning Inference: Evaluate the model's performance on a few examples before fine-tuning.
    Dataset Preparation: Load, preprocess, and tokenize the SQuAD v2 dataset.
    Fine-Tuning: Use the Seq2SeqTrainer from Hugging Face Transformers with LoRA adapters to fine-tune the model.
    Model Saving: Save the fine-tuned model and tokenizer.
    Inference & Evaluation: Load the saved model, perform inference on new examples, and evaluate performance using F1 and Exact Match scores.

Requirements

    Python 3.7+
    PyTorch
    Hugging Face Transformers
    Hugging Face Datasets
    PEFT
    bitsandbytes
    tensorboard
    accelerate
    numpy, pandas

Installation

Install the required packages via pip:

pip install torch transformers datasets peft bitsandbytes tensorboard accelerate

Project Structure

    Quantization
    Configure the BitsAndBytes quantization settings to load the model in 4-bit mode.

    Model Initialization
    Initialize the FLAN-T5 model and tokenizer using the quantization configuration.

    Model Freezing and Gradient Checkpointing
    Freeze the model parameters and enable gradient checkpointing for memory efficiency during training.

    Inference Before Fine-Tuning
    Test the model on sample inputs to observe its pre-fine-tuning performance.

    Helper Functions
    Define functions for inference, prompt creation, and parameter counting.

    Dataset Preparation
    Load the SQuAD v2 dataset, clean and preprocess the data, and tokenize both the context and answers.

    Fine-Tuning with LoRA and Quantization
    Fine-tune the model using LoRA adapters via the Seq2SeqTrainer. The training arguments include batch size auto-finding, learning rate, warmup steps, and more.

    Saving the Model
    Save the fine-tuned model and tokenizer for future use or deployment.

    Model Inference and Evaluation
    Load the saved model, perform inference on new examples, and evaluate the model’s performance using F1 and Exact Match scores.

How to Run

    Set Up Environment:
    Ensure all dependencies are installed and set the necessary environment variables (e.g., CUDA devices, disabling WandB if needed).

    Run the Notebook:
    Open and execute the provided Jupyter Notebook in sequential order to:
        Configure quantization.
        Initialize and prepare the model.
        Preprocess the dataset.
        Fine-tune the model.
        Save and evaluate the model.

    Inference:
    After training, run the inference cells to test the model on new context and question pairs.

Results

    The fine-tuned model achieves competitive F1 and Exact Match scores on the SQuAD v2 validation set.
    Memory usage is significantly reduced due to quantization, making the approach more efficient for resource-constrained environments.

Conclusion

This project demonstrates an end-to-end workflow for fine-tuning a pre-trained FLAN-T5 model using LoRA and quantization techniques. The approach shows promising results on a real-world question answering task and can be extended to other datasets and applications.
