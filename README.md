# Casual Conversation Chatbot (Chit-Chat BOT)

This project is a part of the **Natural Language Processing and GenAI** (AAI-520) course in [the Applied Artificial Intelligence Master Program](https://onlinedegrees.sandiego.edu/masters-applied-artificial-intelligence/) at [the University of San Diego (USD)](https://www.sandiego.edu/). 

-- **Project Status: COMPLETED**

## Project Overview
This project involves developing a generative chatbot using **LSTM** to handle multi-turn conversations and adapt to different user intents. The chatbot will be trained on the **Cornell Movie Dialogues Corpus**, which provides diverse and engaging conversational data. The final output will be a user-friendly web interface where users can interact with the chatbot.

## Objectives
- Train a generative chatbot capable of:
  - Understanding and maintaining context across multiple turns.
  - Responding coherently to a variety of topics.
  - Providing an interactive and accessible experience for users via a web interface.

## Dataset
The chatbot will be trained on the **[Cornell Movie Dialogues Corpus](https://www.kaggle.com/datasets/rajathmc/cornell-moviedialog-corpus)**, which contains:
- **220,000 conversational exchanges** between **10,000+ movie characters**.
- Conversations from over **600 movies**, providing a wide range of contexts, informal speech, and different styles of interaction.

## Model and Architecture

I experimented with three models: **GPT-2**, **T5-small**, and **LSTM**. Each model was chosen for its strengths in handling natural language processing tasks, but adjustments had to be made based on hardware limitations (16 GB RAM).

**Initial Attempts:**

- **GPT-2** and **T5-small** from Hugging Face's Transformers library were the initial choices for their ability to handle complex conversational tasks. However, these models required significant computational power, which posed challenges on the available hardware.

- To address these challenges, several techniques were used:
  - **PEFT LoRA (Parameter Efficient Fine-Tuning)**: Applied to fine-tune GPT-2 and T5-small with fewer parameters, reducing memory usage and making training more efficient.
  - **Smaller Sampled Dataset**: The dataset was down-sampled to fit within the hardware constraints while still being representative enough for the task.
  - **AdamW Optimizer**: Used to improve convergence with weight decay, enhancing training stability.
  - **WandB (Weights and Biases)**: Integrated to monitor model performance in real-time, helping track experiments and hyperparameter tuning.

- Despite these efforts, the performance of **GPT-2** and **T5-small** was limited by the available hardware, leading to long training times and suboptimal results.

**Final Model Selection: LSTM Architecture**

After facing these challenges, I ultimately selected an **LSTM (Long Short-Term Memory)** network for its efficiency and suitability given the hardware constraints. The LSTM model allowed for faster training while maintaining competitive performance for multi-turn conversations.

**LSTM Model with Attention Mechanism and GloVe:**

- **Embedding Layer**: Transforms input words (represented as integers) into 300-dimensional dense vectors using pre-trained GloVe embeddings.
- **Bidirectional LSTM Layer**: A bidirectional LSTM layer with 256 units is used to capture context from both the forward and backward directions of the input sequence
- **Attention Layer**: An Attention Mechanism is introduced to allow the model to focus on the most relevant parts of the input sequence, improving its ability to understand the nuances of conversation
- **Second LSTM Layer**: This additional LSTM layer, with 128 units, further processes the output from the attention layer. Another 30% dropout is applied for regularization, followed by batch normalization to stabilize the training process.
- **Dropout Layers**: Used for regularization to prevent overfitting, with a dropout rate of 0.3.
- **Model Compilation**: The model is compiled with the Adam optimizer and sparse categorical crossentropy as the loss function, suitable for multi-class classification tasks like next-word prediction. Gradient clipping (with clipnorm=1.0) is applied to prevent exploding gradients and ensure stable training. Accuracy is used as the evaluation metric.
- **Dense Layer**: A 64-unit fully connected layer with ReLU activation to introduce non-linearity.
- **Output Layer**: A softmax output layer with a size equal to the vocabulary, predicting the next word in the conversation sequence.
- **Data Generator**: A custom **DataGenerator** was implemented to handle large datasets efficiently by loading and processing the data in batches during training, making the model fit within hardware constraints.
- **ReduceLROnPlateau**: To ensure efficient training, ReduceLROnPlateau is applied to adjust the learning rate when the validation loss plateaus. When the validation loss stops improving for 3 consecutive epochs, the learning rate is reduced by a factor of 0.2 (i.e., it will decrease by 20%). This allows the model to fine-tune and converge better towards optimal solutions. The minimum learning rate is set to 0.0001.
- **Early Stopping**: To prevent overfitting, early stopping is applied, monitoring validation loss. If the validation loss doesn’t improve for 3 consecutive epochs, training stops, and the best model weights are restored.
  

## Future Improvements


## Contributing
Contributions are welcome for future improvements after the initial development phase.

## License
This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for more details.

## Acknowledgements
* A special thanks to **Professor Andrew Van Benschoten, Ph.D.**, for his invaluable guidance and support throughout this class/project.
* The **TensorFlow and PyTorch** communities for their work on deep learning frameworks, including the implementation of LSTM models.
* **Streamlit** for offering an easy-to-use platform to deploy this project as an interactive web app.
* **The Cornell Movie Dialogues Corpus** for providing the dataset that made this project possible.

