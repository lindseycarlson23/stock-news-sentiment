# Stock News Sentiment & Summarization

## 1. Description
This project develops an AI-driven system for a (hypothetical) investment startup to analyze daily stock news. It consists of two main components:
1.  **Sentiment Classifier:** A machine learning model that analyzes news articles to gauge market sentiment (Positive, Neutral, Negative).
2.  **Weekly Summarizer:** A large language model (LLM) that processes a week's worth of news and generates a summary of the top positive and negative events.

## 2. Problem Statement
With an ever-rising number of news articles and opinions, an investment startup aims to leverage artificial intelligence to address the challenge of interpreting stock-related news and its impact on stock prices.

As a member of the Data Science and AI team, I was tasked with analyzing the data, developing an AI-driven sentiment analysis system, and summarizing the news at a weekly level. This system is designed to empower financial analysts with actionable insights, leading to more informed investment decisions.

## 3. Data
The dataset (`stock_news.csv`) contains historical daily news for a specific company, along with its daily stock price data.

### Data Dictionary
| Column | Description |
| --- | --- |
| `Date` | The date the news was released |
| `News` | The content of news articles |
| `Open` | The stock price (in \$) at the beginning of the day |
| `High` | The highest stock price (in \$) reached during the day |
| `Low` | The lowest stock price (in \$) reached during the day |
| `Close` | The adjusted stock price (in \$) at the end of the day |
| `Volume` | The number of shares traded during the day |
| `Label` | The sentiment polarity of the news content (1: positive, 0: neutral, -1: negative) |

## 4. Tools & Technologies
* **Python 3**
* **Data Analysis:** `pandas`, `numpy`, `matplotlib`, `seaborn`
* **NLP Preprocessing:** `nltk` (stopwords, PorterStemmer), `re` (regex)
* **Word Embeddings:** `gensim` (Word2Vec, GloVe), `sentence-transformers` (all-MiniLM-L6-v2)
* **Machine Learning:** `scikit-learn` (RandomForestClassifier, GridSearchCV), `torch`
* **Summarization (LLM):** `llama-cpp-python`, `transformers` (Llama-2-13B-chat-GGUF)

## 5. Methodology

### Part 1: Sentiment Analysis
The goal was to build a classifier to predict the `Label` column based on the `News` text.

1.  **Exploratory Data Analysis (EDA):** Analyzed the distribution of variables, word counts, and the correlation between sentiment labels and stock price changes (`Price_Change` = `Close` - `Open`). The dataset was found to be imbalanced, with a majority of "Neutral" labels.
2.  **Text Preprocessing:** A custom function (`preprocess_text`) was created to clean the news text by converting to lowercase, removing special characters/numbers, removing stopwords, and applying Porter stemming.
3.  **Model Iteration 1 (Word2Vec/GloVe):**
    * Generated document-level embeddings using averaged **Word2Vec** and **GloVe** vectors.
    * Trained several `RandomForestClassifier` models (default, class-weighted, and hyperparameter-tuned).
    * *Result:* These models performed poorly, struggling to distinguish between classes due to the data imbalance and the simplicity of averaged embeddings.
4.  **Model Iteration 2 (SentenceTransformer):**
    * To capture more semantic meaning, embeddings were generated using the **`all-MiniLM-L6-v2`** SentenceTransformer model.
    * A `RandomForestClassifier` was trained on these more sophisticated embeddings.

### Part 2: Weekly News Summarization
1.  **Data Aggregation:** The news articles were grouped by week.
2.  **LLM Prompting:** The **Llama-2 13B (GGUF)** model was used for summarization. A detailed prompt was engineered instructing the model to act as an AI analyst and return a JSON object containing the top positive and negative events for the week that would likely impact stock prices.

## 6. Results
* **Sentiment Model:** The final `RandomForestClassifier` (using SentenceTransformer embeddings) achieved **68% accuracy** and a **0.67 weighted F1-score** on the test set, proving effective at distinguishing between positive, neutral, and negative news.
* **Summarizer:** The Llama-2 model successfully generated structured JSON summaries of weekly news, providing analysts with concise, actionable insights.

## 7. How to Run
1.  Clone the repository:
    ```sh
    git clone [https://docs.github.com/en/repositories/creating-and-managing-repositories/about-repositories](https://docs.github.com/en/repositories/creating-and-managing-repositories/about-repositories)
    ```
2.  Install the required dependencies:
    ```sh
    pip install -U sentence-transformers gensim transformers tqdm pandas numpy matplotlib seaborn nltk scikit-learn torch
    pip install llama-cpp-python==0.1.85
    ```
3.  Download the Llama-2 GGUF model (see notebook for details on `hf_hub_download`).
4.  Open and run the cells in the Jupyter Notebook (`.ipynb`) file.
