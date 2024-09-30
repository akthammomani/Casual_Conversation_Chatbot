# Generative Chatbot for Multi-Turn Conversations

## Project Overview
This project involves developing a generative chatbot using **GPT-2** to handle multi-turn conversations and adapt to different user intents. The chatbot will be trained on the **Cornell Movie Dialogues Corpus**, which provides diverse and engaging conversational data. The final output will be a user-friendly web interface where users can interact with the chatbot.

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
- Starting with **GPT-2 (124M parameters)** from Hugging Face's Transformers library.
- Adjustments will be made to accommodate limited hardware (16 GB RAM), such as using:
  - **Smaller batch sizes**.
  - **Gradient accumulation**.
  - **Mixed precision training**.

- **Model Flexibility**: Depending on the challenges faced, other models may be considered to better suit the available resources.

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
This project is licensed under the **MIT License**. See the `LICENSE` file for more information.

## Acknowledgements
- **OpenAI** for providing **GPT-2**.
- **Hugging Face** for their excellent **Transformers** library.
- The **Cornell Movie Dialogues Corpus** for the dataset.

