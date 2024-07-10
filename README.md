# NLP Tasks with Transformers and BiLSTM

This repository demonstrates the implementation of two advanced Natural Language Processing (NLP) tasks: Text Summarization using BART and Named Entity Recognition (NER) using BiLSTM. These projects showcase the power of modern NLP techniques in solving complex language understanding problems.

## Project Overview

1. Text Summarization with BART
2. Named Entity Recognition with BiLSTM

## 1. Text Summarization with BART

Text summarization is the task of condensing a longer piece of text into a concise summary while retaining the main ideas. This project uses the BART model, which excels at generative tasks like summarization.

### Dataset
We use the DialogSUM dataset, containing 13,460 dialogues covering various daily-life topics. This diverse dataset challenges the model to understand and summarize conversations effectively.

### Model
BART (Bidirectional and Auto-Regressive Transformers) is a powerful sequence-to-sequence model that combines the bidirectional encoder of BERT with the auto-regressive decoder of GPT. We use the pretrained 'facebook/bart-large-xsum' model as our starting point.

### Implementation Details
Our implementation follows these key steps:
- Tokenization using BartTokenizer to prepare the data for the model
- Data collation with DataCollatorForSeq2Seq to handle variable-length sequences
- Training using Seq2SeqTrainer, which simplifies the training process
- Evaluation using ROUGE scores and BLEU score to assess summary quality

We've carefully configured training arguments and implemented custom evaluation metrics to ensure optimal performance.

## 2. Named Entity Recognition with BiLSTM

Named Entity Recognition involves identifying and classifying named entities (e.g., person names, organizations, locations) in text. Our approach uses a Bidirectional LSTM model, which can capture context from both directions in a sequence.

### Dataset
We use a dataset derived from the GMB corpus, containing over a million tagged entities. This rich dataset allows our model to learn a wide variety of entity types and contexts.

### Model
Our model architecture consists of:
- An embedding layer initialized with Word2Vec embeddings
- Three Bidirectional LSTM layers to capture context
- A Dense layer for feature extraction
- A Conditional Random Field (CRF) layer for optimal tag sequence prediction

This combination allows the model to understand context and maintain consistency in entity labeling.

### Implementation Details
The implementation involves several crucial steps:
1. Data preprocessing: We tokenize the text, create word and tag indices, and prepare word embeddings.
2. Dataset preparation: This includes converting tokens to indices, padding sequences, and one-hot encoding of tags.
3. Model building: We construct the BiLSTM-CRF model using Keras functional API.
4. Training: We use callbacks like ModelCheckpoint and EarlyStopping for efficient training.
5. Evaluation: We assess the model using accuracy metrics and a confusion matrix.


## Results
We evaluate our models using standard NLP metrics:
- For Text Summarization: ROUGE scores measure the overlap between generated and reference summaries, while BLEU scores assess the quality of the generated text.
- For Named Entity Recognition: We report accuracy scores and provide a confusion matrix to show the model's performance across different entity types.


## Future Work
We plan to extend this project by fine-tuning the PEGASUS model for text summarization using the DialogSUM dataset. This will allow us to compare different architectures and potentially improve our summarization results.
