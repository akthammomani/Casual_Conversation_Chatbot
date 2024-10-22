
# Model and Architecture

I experimented with two models: **GPT-2** and **T5-small**. Each model was chosen for its strengths in handling natural language processing tasks, but adjustments had to be made based on hardware limitations (16 GB RAM).

**Initial Attempts:**

- **GPT-2** (medium as well) and **T5-small** from Hugging Face's Transformers library were the initial choices for their ability to handle complex conversational tasks. However, these models required significant computational power, which posed challenges on the available hardware.

- To address these challenges, several techniques were used:
  - **PEFT LoRA (Parameter Efficient Fine-Tuning)**: Applied to fine-tune GPT-2 and T5-small with fewer parameters, reducing memory usage and making training more efficient.
  - **Smaller Sampled Dataset**: The dataset was down-sampled to fit within the hardware constraints while still being representative enough for the task.
  - **AdamW Optimizer**: Used to improve convergence with weight decay, enhancing training stability.
  - **WandB (Weights and Biases)**: Integrated to monitor model performance in real-time, helping track experiments and hyperparameter tuning.

- Despite these efforts, the performance of **GPT-2** and **T5-small** was limited by the available hardware, leading to long training times and suboptimal results.
