GPT-2 Fine-Tuning Documentation for Poetry Generation
This document provides a complete explanation of the GPT-2 fine-tuning process for poetry generation. It walks through the full pipeline from data preprocessing to training and saving the model, and includes explanations for all major parameters and configurations used during the training process. This guide is intended for future reference and reproducibility.
Step 1: Environment Preparation and Imports
To ensure smooth training, especially on machines with GPUs, we enable a setting that allows CUDA memory to be allocated flexibly. We also import required libraries: Hugging Face Transformers for model training, Datasets for handling data, Torch for the deep learning backend, and Pandas for reading the CSV data.
Step 2: Load and Extend the GPT-2 Model
We load a pre-trained GPT-2 model using the Hugging Face library. Since GPT-2 is not trained on poem-specific tokens, we manually add special tokens like <|startofpoem|>, <|endofpoem|>, and <|pad|>. This helps the model recognize the structure of poems and handle padding correctly.
Step 3: Dataset Preparation and Formatting
We use a CSV file containing poems along with their metadata such as title, poet name, and tags. Short poems (less than 50 characters) are removed to avoid noise. Missing values are replaced with empty strings to maintain format consistency.
Each row in the dataset is then formatted into a structured string using special tokens. Here's an example of a single preprocessed entry:
<|startofpoem|>
Title: A Summer Day
Poet: John Doe
Tags: Nature, Sunshine

The sun shines bright on a field of gold,
With stories that the breeze has told,
Each petal bends in morning grace,
Touched gently by the sky's embrace.
<|endofpoem|>
Step 4: Tokenization
Each formatted poem is converted into tokens using the GPT-2 tokenizer. These tokens are numerical representations of words or symbols. All input sequences are either padded or truncated to a maximum length of 768 tokens, which balances training time and memory usage.
Step 5: Train-Validation Split
The full dataset is split into two parts: 80% for training and 20% for validation. This allows the model to learn from the majority of data while being evaluated on unseen data to check for overfitting or underfitting.
Step 6: Data Collator
We use a data collator that dynamically pads input sequences and sets up data for causal language modeling. Unlike masked language models (MLM), GPT-2 is trained to predict the next token, not random masked tokens.
Step 7: Training Configuration
Below are the training arguments used and why they were chosen:
• output_dir: Directory to save model checkpoints and logs.
• overwrite_output_dir: Ensures the training can overwrite previous runs.
• num_train_epochs: Set to 10 so the model gets multiple passes over data for improved learning.
• per_device_train_batch_size: Batch size of 4 per GPU. Adjusted for memory limits.
• gradient_accumulation_steps: Set to 2 to simulate a batch size of 8, helping with stability and memory use.
• save_steps: Save model every 1000 steps to avoid losing progress.
• save_total_limit: Limit to 2 recent checkpoints to save disk space.
• logging_dir: Directory to save training logs.
• logging_steps: Logs written every 100 steps for visibility into training.
• fp16: Enables mixed precision training if supported by GPU (improves speed and reduces memory).
• warmup_steps: Uses 200 steps to gradually increase the learning rate, preventing unstable jumps early in training.
• weight_decay: Applies slight regularization to avoid overfitting.
• learning_rate: Set to 3e-5, which is a moderate and stable rate for fine-tuning pretrained models.
Step 8: Model Training
We use Hugging Face’s Trainer class to perform training. It handles batching, evaluation, checkpointing, and logging. The training process involves feeding tokenized sequences into the model, calculating loss, and adjusting weights using backpropagation.
Step 9: Saving the Trained Model
Once the training is complete, the final model and tokenizer are saved in a directory. These can later be reloaded for text generation or further fine-tuning without repeating the training process from scratch.
 Conclusion
This document serves as a complete guide for setting up, training, and saving a fine-tuned GPT-2 model for poem generation. It includes detailed explanations for each step and parameter to ensure reproducibility and clarity for future work.
