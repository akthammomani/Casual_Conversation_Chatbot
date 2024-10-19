# **Data Wrangling: Casual Conversation Chatbot (Chit-Chat Bot)**

* **Group 8:** Aktham Almomani
* **Course:** Natural Language Processing and GenAI (AAI-520-04) / University Of San Diego
* **Semester:** Winter 2024

<center>
    <img src="https://github.com/akthammomani/AI_powered_heart_disease_risk_assessment_app/assets/67468718/2cab2215-ce7f-4951-a43a-02b88a5b9fa9" alt="wrnagling">
</center>


## **Dataset**

**[The Cornell Movie Dialogues Dataset](https://www.kaggle.com/datasets/rajathmc/cornell-moviedialog-corpus)** is a well-known and widely-used dataset for building conversational models, particularly chatbots. It contains dialogues from over 600 movies, with a rich collection of character interactions, making it an excellent resource for training dialogue-based models.

**Key Features of the Dataset:**
* **Multi-Turn Conversations:** The dataset includes over 220,000 conversational exchanges between pairs of movie characters. These exchanges provide valuable data for modeling multi-turn conversations, which are essential for creating chatbots that can engage in longer, more meaningful conversations.

* **Diverse Dialogue Styles:** The dataset spans many genres and includes conversations from a wide range of characters, settings, and speaking styles, from casual and humorous dialogue to intense or emotional exchanges. This diversity is ideal for training models that can handle different conversational tones and contexts.

* **Structured Data:**

   * `movie_lines.txt:` Contains individual lines of dialogue, along with metadata such as the character and movie from which the line originates.
   * `movie_conversations.txt:` Links together individual lines into conversations, providing a structured flow of multi-turn dialogue.

* **Movie Metadata:** While the primary focus is on dialogue, the dataset also includes metadata such as movie titles, character names, and years of release. These can be useful for building context-aware systems or for generating trivia-based responses, though they are not necessary for general chit-chat functionality.

**Dataset Composition:**
* Number of Conversations: Over `83,000` multi-turn conversations.
* Number of Dialogue Lines: Around `220,000` individual lines of dialogue.
* Character and Movie Diversity: Dialogue from over `10,000` characters in `617` different movies, offering a wide variety of conversational topics and styles.

**Why This Dataset?**
The Cornell Movie Dialogues Dataset is particularly suitable for training generative chatbots because:

* **Rich Multi-Turn Structure:** It offers high-quality dialogue exchanges that are perfect for building models capable of sustaining coherent conversations.
* **Diverse Conversations:** The wide range of conversational tones allows models to learn different ways of responding, making the chatbot more versatile in its responses.
* **Well-Structured Data:** The structured format of conversations and lines makes it easier to process, tokenize, and use for training conversational models.

By using this dataset, we can build a chatbot that is capable of handling casual conversations, answering questions, and maintaining coherent multi-turn dialogue, making it ideal for applications like customer support, virtual assistants, and entertainment bots.

## **Data Wrangling & Preprocessing**

In this notebook, we perform the critical steps of data wrangling and preprocessing to prepare the **Cornell Movie Dialogues Dataset** for training a conversational chatbot. This dataset was imported from **Kaggle**, and it contains dialogues from a variety of movies, offering a rich set of conversation pairs for building a chatbot capable of engaging in casual conversations.

**Data Import:**

The dataset was sourced from **Kaggle** and consists of multiple text files including:

* `movie_lines.txt:` Contains individual lines of dialogue along with metadata.
* `movie_conversations.txt:` Provides conversation groupings, linking individual lines to create multi-turn dialogues.

We imported these files into Python and structured the data for efficient processing. The data was then loaded into a Pandas DataFrame for easier manipulation.

**Data Cleaning and Column Selection:**

Since the focus of our chatbot is **casual conversation** rather than movie-specific trivia or character emulation, we discarded irrelevant columns such as:

* **Movie Titles:** Not essential for casual conversations.
* **Character Names:** Not required for general chit-chat.
* **Years:** Historical context is unnecessary in a conversational chatbot.

After removing these unnecessary columns, we focused on the core **dialogue text**, which will serve as the foundation for training the chatbot.

**Preprocessing Pipeline:**

The preprocessing pipeline is designed to clean and standardize the conversational data for input into the model. The steps include:

* **Expanding Contractions:** Common English contractions (e.g., can't to cannot) were expanded to ensure the text is uniform.
* **Removing Non-Dialogue Elements:** Stage directions and non-verbal actions, such as descriptions in parentheses or square brackets, were removed to keep only the spoken parts.
* **Preserving Proper Nouns:** Using natural language processing (NLP), we ensured that proper nouns such as movie titles, places, and names were preserved in their original form, while the rest of the text was lowercased.
* **Normalizing Numbers:** Numbers and years were normalized by replacing them with tokens such as <NUMBER> and <YEAR>.
* **Truncating Long Utterances:** Very long dialogue lines were truncated to a maximum word length to maintain coherence.
* **Final Cleanup:** The text was cleaned of any remaining unwanted characters and extra spaces.
* 
Each step in this pipeline helps standardize the input text and removes noise that could negatively impact the chatbot's performance. After applying the cleaning pipeline, the data was stored in a **cleaned DataFrame** for subsequent processing and model training.

By applying this preprocessing pipeline, we ensure the text is well-structured, uniform, and ready for the next steps in building our chatbot.
