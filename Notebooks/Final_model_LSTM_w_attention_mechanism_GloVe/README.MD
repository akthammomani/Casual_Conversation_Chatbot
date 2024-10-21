
# **Final Model: LSTM Model with Attention Mechanism and GloVe**

## Introduction: 
For this project, we selected **LSTM (Long Short-Term Memory)** as the final architecture for building our chatbot. LSTM is particularly well-suited for handling sequential data, such as conversations, due to its ability to retain information over long sequences and handle dependencies across time steps. This makes it an excellent choice for our chit-chat bot, where maintaining conversational context is crucial across multiple turns.

**Why LSTM is a Good Choice:**

* Memory Retention: Unlike traditional RNNs, LSTMs can remember long-term dependencies in a conversation, which is essential for context-sensitive responses.
* Handling Long Conversations: LSTMs manage the vanishing gradient problem better than basic RNNs, allowing them to perform well with longer conversations where context from earlier exchanges matters.
* Efficiency: LSTM’s cell states allow it to selectively forget or remember information, making it more efficient for training on sequence data.

**LSTM vs. RNN:**

* **RNN (Recurrent Neural Networks):** While RNNs are also used for sequence-based tasks, they have limitations due to the vanishing gradient problem, where older information tends to be lost during backpropagation, which can be detrimental to multi-turn conversations in a chatbot.

* **LSTM:** Overcomes these issues by introducing gates (input, forget, and output gates) that allow it to control the flow of information through the network, retaining important parts of the conversation across longer sequences.

Given these advantages, LSTM was chosen as the final architecture for our model, enabling the chatbot to handle conversational context and provide coherent multi-turn responses.

## Data Preparation:

### Filtering Conversations by Length

To enhance data quality and ensure that our model trains on meaningful interactions, we filtered the conversations based on their length. We kept only those conversations that have more than `2 words` and less than `24 words`. This step removes extremely short exchanges that lack context and excessively long conversations that may introduce noise.

### Removing Rare Words

To reduce vocabulary size and focus on more commonly used language, we removed rare words that occur less than 5 times in the dataset. Eliminating these infrequent words helps the model to concentrate on patterns within the core vocabulary, improving learning efficiency and performance.

### Assessing Conversation Length

Understanding the length of the conversations in our dataset was crucial for designing the LSTM architecture. We assessed the distribution of conversation lengths to ensure that the model's input size was appropriate.

Here’s a visual representation of the distribution of conversation lengths:

![conv_length](https://github.com/user-attachments/assets/13acb4ed-5664-4035-8320-b23893ba9cfe)

Mean conversation length: 10.87 words
Median conversation length: 7 words
90th percentile: 24 words
Based on this distribution, we chose a maximum sequence length of 24 for the model, as it covers the vast majority of conversations in the dataset without excessively truncating or padding sequences.

### Assessing Sequence Length

Understanding the length of the tokenized sequences was crucial for designing our LSTM architecture and ensuring efficient model training. After tokenizing the conversation data—including special tokens like <`CONV_START`> and <`CONV_END`>—we assessed the distribution of sequence lengths to determine an appropriate maximum sequence length (`max_sequence_length`).

To balance the need for sufficient contextual information with computational efficiency, we selected the `90th percentile of sequence lengths` as our max_sequence_length. In our dataset, this value was `54 tokens`, meaning that 90% of our sequences were 54 tokens or shorter.


### Preparing Conversation Sequences with Context

To enhance the chatbot's ability to understand and generate contextually relevant responses, we prepare conversation sequences that encapsulate the entire dialogue flow. By grouping messages belonging to the same conversation and maintaining their chronological order, we provide the model with the necessary context to learn from previous turns in the conversation.

**Add special tokens:**

- <`CONV_START`> at the beginning to signify the start of a conversation.
- <`CONV_END`> at the end to indicate the conversation's conclusion.


## LSTM Model Architecture
The LSTM model was built to handle sequential input data (conversations) and predict the next word in the conversation. Here’s an overview of the architecture:

- **Embedding Layer**: Transforms input words (represented as integers) into `300-dimensional dense vectors` using `pre-trained GloVe embeddings`.
- **Bidirectional LSTM Layer**: A bidirectional LSTM layer with `256 units` is used to capture context from both the forward and backward directions of the input sequence
- **Attention Layer**: An Attention Mechanism is introduced to allow the model to focus on the most relevant parts of the input sequence, improving its ability to understand the nuances of conversation
- **Second LSTM Layer**: This additional LSTM layer, with `128 units`, further processes the output from the attention layer. Another `30% dropout` is applied for regularization, followed by batch normalization to stabilize the training process.
- **Dropout Layers**: Used for regularization to prevent overfitting, with a dropout rate of 0.3.
- **Model Compilation**: The model is compiled with the Adam optimizer and sparse categorical crossentropy as the loss function, suitable for multi-class classification tasks like next-word prediction. Gradient clipping (with clipnorm=1.0) is applied to prevent exploding gradients and ensure stable training. Accuracy is used as the evaluation metric.
- **Dense Layer**: A 64-unit fully connected layer with ReLU activation to introduce non-linearity.
- **Output Layer**: A softmax output layer with a size equal to the vocabulary, predicting the next word in the conversation sequence.
- **Data Generator**: A custom **DataGenerator** was implemented to handle large datasets efficiently by loading and processing the data in batches during training, making the model fit within hardware constraints.
- **ReduceLROnPlateau**: To ensure efficient training, ReduceLROnPlateau is applied to adjust the learning rate when the validation loss plateaus. When the validation loss stops improving for 3 consecutive epochs, the learning rate is reduced by a factor of 0.2 (i.e., it will decrease by 20%). This allows the model to fine-tune and converge better towards optimal solutions. The minimum learning rate is set to 0.0001.
- **Early Stopping**: To prevent overfitting, early stopping is applied, monitoring validation loss. If the validation loss doesn’t improve for 3 consecutive epochs, training stops, and the best model weights are restored.


## DataGenerator for Efficient Training:
To handle the large dataset efficiently, we implemented a DataGenerator. This allowed the model to train on batches of data rather than loading the entire dataset into memory at once.

**Key Features of the DataGenerator:**

* **Batch Processing:** The generator splits the dataset into smaller, more manageable batches, reducing the memory load and ensuring that training can proceed efficiently.

* **Shuffling:** The data is shuffled before each epoch to prevent the model from overfitting to specific patterns in the training data.

Here’s the structure of the DataGenerator:

```python
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, input_sequences, target_words, batch_size, max_sequence_length, tokenizer):
        self.input_sequences = input_sequences
        self.target_words = target_words
        self.batch_size = batch_size
        self.max_sequence_length = max_sequence_length
        self.tokenizer = tokenizer
    
    def __len__(self):
        return int(np.ceil(len(self.input_sequences) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_sequences = self.input_sequences[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_targets = self.target_words[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        # Convert to appropriate format for model input
        batch_sequences = pad_sequences(batch_sequences, maxlen=self.max_sequence_length, padding='post')
        
        return np.array(batch_sequences), np.array(batch_targets)

```
This DataGenerator ensured that training could be carried out even with memory constraints, making the process more scalable and efficient.


## LSTM LOSS & Accuracy

![final_lstm](https://github.com/user-attachments/assets/b0621778-0a6b-48ce-8025-b8687420fa70)

**Summary Highlights:**

* **Training Accuracy:** Started at `58.81%` in the first epoch and increased steadily, reaching `62.15%` by the fifth epoch. This shows that the model is learning and improving its predictions on the training data.

* **Validation Accuracy:** Although the training accuracy improved, the validation accuracy remained stable, fluctuating between `60.10`% and `60.57%`. This suggests that while the model learns the training data, its ability to generalize to unseen data is limited.

* **Training Loss:** Decreased from `2.5829` to `2.3513`, indicating that the model's predictions on the training data are becoming more accurate with each epoch.

* **Validation Loss:** Remained fairly stable, starting at `2.3555` and ending at `2.4076`. This suggests that the model may be struggling to improve its performance on the validation set, potentially indicating overfitting or insufficient capacity to generalize.

* **Accuracy Gap:** A small gap between the training and validation accuracies (~`1.58%`), which could indicate mild overfitting. The model performs slightly better on the training set than on the validation set.

**Key Observations:**

* The **training accuracy** is improving, but the validation accuracy is relatively stable, suggesting that the model may not be learning new information beyond a certain point.
* The **training loss** is reducing steadily, but the validation loss is not decreasing at the same rate, which could indicate the model's limitations in generalizing well to unseen data.
* **Early Stopping** is a good safeguard here, as the model might not benefit significantly from further training beyond a certain point, as evidenced by the flattening validation performance.

## Human-based Evaluation

Based on the human evaluation of the chit-chat bot, here are the key highlights:

* **Limited Coherency:** The bot struggles to generate meaningful or coherent responses. For instance, when asked common questions like "how are you?" or "what's your name?" it replies with non-relevant terms such as "not" or "the."

* **Repetitive Responses:** The bot tends to provide similar responses like "start" or "the," which suggests it may not fully understand the context of the conversation or have a rich enough response generation capability.

* **Lack of Contextual Awareness:** It appears that the bot is not capturing the context of the dialogue or understanding follow-up questions. For example, it doesn’t respond well to prompts like "give me a joke" or "best comedy movie?"

* **Possible Over-reliance on Specific Tokens:** The bot's responses seem heavily influenced by specific tokens like "start" or "the," suggesting it might have an over-reliance on frequent tokens in the training data, potentially due to class imbalance.

* **Improvement Needed in Decoding Strategy:** The model might benefit from improvements in the decoding strategy, such as beam search, to explore more meaningful word sequences and provide better answers.

## BLEU & ROUGE Evaluation

* **BLEU and ROUGE Scores:** The BLEU and ROUGE scores consistently show zero, indicating that the chatbot responses do not match the reference responses. 

* **Repetitive Responses:** The chatbot outputs repetitive responses like "the" and "i," suggesting that the model may not be learning rich or diverse conversational patterns. This could be due to an imbalanced dataset, insufficient training, or issues with tokenization and vocabulary size.


## Future Considerations

* **Adding an Attention Mechanism:** Incorporating an attention mechanism could significantly enhance the model’s ability to focus on relevant parts of the input sequence, especially for longer conversations. This would likely improve both the model’s performance and its ability to generate more contextually appropriate responses in tasks like chit-chat.

* **Increasing the Number of Epochs:** If the model is still improving towards the end of training, consider extending the number of epochs. This could allow the model to learn more complex patterns, provided that early stopping continues to monitor for overfitting.

* **Hyperparameter Tuning:** To optimize the model further, hyperparameter tuning can be done on the following:

   * **Learning rate:** Experimenting with different learning rates can help the model converge more effectively.
   * **Batch size:** Adjusting the batch size may improve model performance and training speed.
   * **LSTM Units:** Tuning the number of LSTM units in each layer could help the model capture more nuanced features in the data.
   * **Dropout Rate:** Fine-tuning the dropout rates may help improve generalization and prevent overfitting.
   * **Using Pre-trained Embeddings:** Another improvement could be to use pre-trained word embeddings (like GloVe or Word2Vec) instead of training embeddings from scratch. This could provide the model with richer word representations and potentially improve its accuracy.

* **Expanding the Dataset:** If more data is available, training the model on a larger, more diverse dataset could improve its ability to generalize to unseen conversations.

* **Ensemble Methods:** In the future, combining different architectures such as CNNs or other types of recurrent networks with LSTMs could potentially enhance performance further, especially when dealing with more complex conversational data.

Implementing these considerations could lead to improved performance, making the model more robust and versatile in its sequence prediction tasks.
