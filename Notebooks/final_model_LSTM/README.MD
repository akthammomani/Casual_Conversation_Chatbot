# **Final Model: LSTM for Chit-Chat Bot**

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

## Data Preparation - Assessing Conversation Length

Understanding the length of the conversations in our dataset was crucial for designing the LSTM architecture. We assessed the distribution of conversation lengths to ensure that the model's input size was appropriate.

Here’s a visual representation of the distribution of conversation lengths:

![conv_length](https://github.com/user-attachments/assets/13acb4ed-5664-4035-8320-b23893ba9cfe)

Mean conversation length: 10.87 words
Median conversation length: 7 words
90th percentile: 24 words
Based on this distribution, we chose a maximum sequence length of 24 for the model, as it covers the vast majority of conversations in the dataset without excessively truncating or padding sequences.

## Dataset Sampling:

Given hardware limitations and the size of our full dataset, we randomly selected 30% of the dataset for training. This random sampling was done to ensure that the reduced dataset remained representative of the overall dataset's conversational diversity.

To achieve this, the following code was used to sample the dataset:

```python
sampled_data = data.sample(frac=0.3, random_state=42)
```
By selecting 30% of the data, we ensured that the training process was manageable while still maintaining the integrity of the dataset, as it includes conversations from all parts of the original dataset.

## LSTM Model Architecture
The LSTM model was built to handle sequential input data (conversations) and predict the next word in the conversation. Here’s an overview of the architecture:

* **Embedding Layer:** Converts input tokens (words) into dense vector representations, capturing semantic relationships between words.

* **Bidirectional LSTM Layer:** A Bidirectional LSTM with `128 units` was used to process the conversation data. The bidirectional nature allows the model to consider context from both previous and future words in the sequence, improving the quality of the predictions.

* **Dropout Layer:** Dropout with a rate of `0.3` was added to prevent overfitting, ensuring that the model generalizes well to unseen data.

* **Dense Layer:** A fully connected layer with `64 units` and ReLU activation was used to introduce non-linearity into the model and help capture more complex patterns in the data.

* **Output Layer:** The final output is a softmax layer that predicts the probability of the next word in the conversation, outputting a vector with a size equal to the vocabulary.


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


## Conclusion

By selecting LSTM as the final architecture, adjusting for conversation length, and using a DataGenerator to handle the large dataset efficiently, the model was able to learn effectively despite hardware limitations. The use of LSTM provided the necessary balance between model complexity and performance, making it ideal for our chit-chat bot, capable of maintaining conversational context over multiple turns.
