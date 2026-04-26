# i22-1891-NLP-Assignment3

Transformer-Based RAG System for Review Understanding and Explanation Generation
Overview

This project implements a complete Retrieval-Augmented Generation (RAG) system from scratch using Transformer-based architectures. The system performs:

Review Understanding (Encoder)
Semantic Retrieval (Retrieval Module)
Explanation Generation (Decoder)

It is trained and evaluated on a large-scale Amazon Reviews dataset containing 39,000 reviews from multiple product categories.

The main objective is to study whether retrieval augmentation improves generation quality, coherence, and grounding compared to a baseline model.

System Architecture

The pipeline consists of three main components:

1. Encoder (Representation Learning)
Custom Transformer encoder (no pretrained models)
Multi-task learning:
Sentiment classification (3 classes)
Review length classification (3 classes)
Generates dense embeddings for retrieval
2. Retrieval Module
Stores encoder embeddings for all training samples
Uses cosine similarity for nearest neighbor search
Retrieves top-k semantically similar reviews
3. Decoder (Explanation Generation)
Custom decoder-only Transformer
Autoregressive text generation
Conditioned on:
Original review
Predicted sentiment
Predicted length class
Retrieved context
Dataset
Total Reviews: 39,000
Categories:
Beauty (13,000)
Cellphone (13,000)
Electronics (13,000)
Label Distribution
Negative: 4,783
Neutral: 3,632
Positive: 30,585
Training Details
Encoder
d_model: 128
heads: 4
layers: 3
optimizer: Adam
learning rate: 3e-4
Decoder
trained for 6 epochs
causal masking implemented manually
sampling: top-k + temperature (0.8)
Key Results
Encoder Performance
Sentiment Accuracy: 58%
Length Accuracy: 81%
Decoder Performance
Final Loss: 2.98
Improved fluency with retrieval
RAG vs Baseline
Model	Performance
Full RAG	Better coherence, lower perplexity
Baseline	Repetitive and less grounded outputs

Retrieval significantly improves explanation quality by grounding generation in similar examples.

Key Observations
Retrieval improves semantic coherence
Best retrieval setting: k = 3
Encoder embeddings are highly meaningful
Baseline model suffers from repetition and weak context
Important Files
Included in Repository:
encoder.py – Transformer encoder implementation
decoder.py – Transformer decoder implementation
train_encoder.py – encoder training pipeline
train_decoder.py – decoder training pipeline
retrieval.py – similarity search module
preprocessing.py – dataset preprocessing
evaluation.py – metrics and evaluation scripts
Large Files (Not Uploaded to GitHub)

Due to size limitations, the following files are NOT included in the repository:

retrieval_index.pkl
best_decoder.pt
Reason:

These files are too large to be uploaded on GitHub and are generated after training.

How to regenerate:

Run the following steps locally:

# Train encoder to regenerate embeddings
python train_encoder.py

# Build retrieval index
python build_retrieval_index.py

# Train decoder
python train_decoder.py

This will recreate:

retrieval_index.pkl
best_decoder.pt
How to Run
# Install dependencies
pip install -r requirements.txt

# Train encoder
python train_encoder.py

# Build retrieval index
python build_retrieval_index.py

# Train decoder
python train_decoder.py

# Evaluate system
python evaluate.py
Visual Results

All plots are generated during training:

Encoder loss curves
Classification metrics
Retrieval similarity analysis
Decoder loss curve
RAG vs baseline comparison

Saved figures:

encoder_loss.png
retrieval_analysis.png
decoder_learning_curves.png
rag_vs_baseline.png
Conclusion

This project demonstrates a full end-to-end RAG pipeline built from scratch, showing that retrieval-augmented systems significantly outperform standalone generative models in:

Coherence
Grounding
Fluency
Interpretability
