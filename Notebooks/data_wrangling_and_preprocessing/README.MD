# **Data Wrangling: Casual Conversation Chatbot (Chit-Chat Bot)**

* **Group 8:** Aktham Almomani
* **Course:** Natural Language Processing and GenAI (AAI-520-04) / University Of San Diego
* **Semester:** Winter 2024


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


## **Importing Dataset**

```python
# Load movie lines and conversations into DataFrames:
def load_data_to_dataframe():
    # Load movie lines:
    lines_data = []
    with open("movie_lines.txt", encoding='utf-8', errors='ignore') as file:
        for line in file:
            line_parts = line.split(" +++$+++ ")
            if len(line_parts) == 5:
                line_id, character_id, movie_id, character_name, text = line_parts
                lines_data.append([line_id, character_id, movie_id, character_name, text.strip()])

    # Create DataFrame for lines:
    lines_df = pd.DataFrame(lines_data, columns=["line_id", "character_id", "movie_id", "character_name", "text"])

    # Load movie conversations:
    conversations_data = []
    with open("movie_conversations.txt", encoding='utf-8', errors='ignore') as file:
        for line in file:
            line_parts = line.split(" +++$+++ ")
            if len(line_parts) == 4:
                character_ids = line_parts[0:2]
                movie_id = line_parts[2]
                conversation_ids = eval(line_parts[3])  # List of line IDs
                conversations_data.append([character_ids, movie_id, conversation_ids])

    # Create DataFrame for conversations:
    conversations_df = pd.DataFrame(conversations_data, columns=["character_ids", "movie_id", "conversation_ids"])

    # Load movie metadata:
    movie_metadata = []
    with open("movie_titles_metadata.txt", encoding='utf-8', errors='ignore') as file:
        for line in file:
            line_parts = line.split(" +++$+++ ")
            if len(line_parts) >= 5:
                movie_id, movie_title, movie_year, movie_rating, movie_genres = line_parts[:5]
                movie_metadata.append([movie_id.strip(), movie_title.strip(), movie_year.strip(), movie_rating.strip(), movie_genres.strip()])

    # Create DataFrame for movies:
    movies_df = pd.DataFrame(movie_metadata, columns=["movie_id", "title", "year", "rating", "genres"])

    return lines_df, conversations_df, movies_df

# Display complete data with conversation IDs, turn IDs, and progress tracking:
def display_conversations_df(conversations_df, lines_df, movies_df):
    conversation_list = []
    
    # Loop through the conversations and assign conversation IDs:
    for convo_id, convo_row in tqdm(enumerate(conversations_df.itertuples(), start=1), total=len(conversations_df), desc="Processing Conversations"):
        convo_lines = []
        for turn_id, line_id in enumerate(convo_row.conversation_ids, start=1):
            # Get the line text from the lines_df
            line_data = lines_df[lines_df['line_id'] == line_id]
            if not line_data.empty:
                text = line_data.iloc[0]['text']
                character_name = line_data.iloc[0]['character_name']
                movie_id = line_data.iloc[0]['movie_id']

                # Get movie title and year from movies_df
                movie_data = movies_df[movies_df['movie_id'] == movie_id]
                if not movie_data.empty:
                    movie_title = movie_data.iloc[0]['title']
                    movie_year = movie_data.iloc[0]['year']
                else:
                    movie_title = "Unknown"
                    movie_year = "Unknown"
                
                # Append conversation with conversation ID, turn ID, and text:
                convo_lines.append([convo_id, turn_id, movie_title, movie_year, character_name, text])
        
        # Add all turns of the conversation to the list:
        conversation_list.extend(convo_lines)

    # Create DataFrame for complete conversations with conversation and turn IDs:
    full_conversations_df = pd.DataFrame(conversation_list, columns=["Conversation ID", "Turn ID", "Movie Title", "Year", "Character", "Text"])
    return full_conversations_df
```


