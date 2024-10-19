# Generative Chatbot for Multi-Turn Conversations

This project is a part of the **Natural Language Processing and GenAI** (AAI-520) course in [the Applied Artificial Intelligence Master Program](https://onlinedegrees.sandiego.edu/masters-applied-artificial-intelligence/) at [the University of San Diego (USD)](https://www.sandiego.edu/). 

-- **Project Status: Ongoing**

## Project Overview
This project involves developing a generative chatbot using **LSTM** to handle multi-turn conversations and adapt to different user intents. The chatbot will be trained on the **Cornell Movie Dialogues Corpus**, which provides diverse and engaging conversational data. The final output will be a user-friendly web interface where users can interact with the chatbot.

## Objectives
- Train a generative chatbot capable of:
  - Understanding and maintaining context across multiple turns.
  - Responding coherently to a variety of topics.
  - Providing an interactive and accessible experience for users via a web interface.

## Features
- **Multi-Turn Conversational Capabilities**: The chatbot can manage and retain context through consecutive user interactions.
- **Web Interface**: Built with **Streamlit** to provide an easy-to-use platform for chatting with the bot.
- **Adaptability**: Future improvements planned for increasing coherence, latency optimization, and expanding the knowledge domain.

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

**LSTM Model Architecture:**

- **Embedding Layer**: Transforms the input tokens into dense vector representations.
- **Bidirectional LSTM Layer**: A bidirectional LSTM with 128 units, allowing the model to capture information from both past and future context.
- **Dropout Layers**: Used for regularization to prevent overfitting, with a dropout rate of 0.3.
- **Dense Layer**: A 64-unit fully connected layer with ReLU activation to introduce non-linearity.
- **Output Layer**: A softmax output layer with a size equal to the vocabulary, predicting the next word in the conversation sequence.
- **Data Generator**: A custom **DataGenerator** was implemented to handle large datasets efficiently by loading and processing the data in batches during training, making the model fit within hardware constraints.

**Model Flexibility**

Throughout the process, model flexibility remained a key consideration, with adjustments made as needed to fit the available resources.


## Challenges and Adaptation Strategy
- **Limited Resources**: The project will be developed on a machine with **16 GB RAM**, which requires careful optimization to ensure successful training and deployment.
- Strategies include:
  - Using **smaller batch sizes** and **gradient accumulation** to fit model training into limited memory.
  - Keeping **options open** to switch to a lighter model if needed.

## Installation
To run this project locally, you need to have Python installed. Follow these steps:

1. Clone the repository:
    ```sh

    ```

2. Install the required libraries:
    ```sh
   
    ```

3. Run the chatbot interface using **Streamlit**:
    ```sh

    ```

## Usage
- Run the **Streamlit app** to interact with the chatbot.
- Users can enter conversational inputs and receive responses from the GPT-2 model in real-time.

## Project Milestones
1. **Week 1**: Research and Dataset Preprocessing.
2. **Week 2**: Model Training and Initial Testing.
3. **Week 3**: Web Interface Development, Testing, and Deployment.

## Future Improvements


## Contributing
Contributions are welcome for future improvements after the initial development phase.

## License
This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for more details.

## Acknowledgements
- **OpenAI** for providing **GPT-2**.
- **Hugging Face** for their excellent **Transformers** library.
- The **Cornell Movie Dialogues Corpus** for the dataset.

