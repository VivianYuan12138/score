import os
import json
import re
from collections import OrderedDict, defaultdict
import openai
from openai import OpenAI

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from config import openai
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize OpenAI API key and client
client = OpenAI(api_key=openai.api_key)
total_token_count = 0

# Load sentiment analysis results from file
def load_sentiment_results(SENTIMENT_RESULTS_PATH):
    if os.path.exists(SENTIMENT_RESULTS_PATH):
        with open(SENTIMENT_RESULTS_PATH, 'r') as f:
            return json.load(f)
    return {}

# Save sentiment analysis results to file
def save_sentiment_results(results, SENTIMENT_RESULTS_PATH):
    with open(SENTIMENT_RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

# Retrieve saved sentiment score for a specific episode
def get_saved_sentiment(episode_number, sentiment_results):
    return sentiment_results.get(str(episode_number))

# Analyze sentiment or retrieve saved result for an episode
def analyze_or_get_sentiment(episode_number, text, sentiment_results):
    saved_result = get_saved_sentiment(episode_number, sentiment_results)
    if saved_result is not None:
        print(f"Retrieved saved sentiment analysis for Episode {episode_number}: {saved_result}")
        return saved_result
    else:
        sentiment_score, token_usage = analyze_sentiment_openai(text)
        sentiment_results[str(episode_number)] = sentiment_score
        print(f"Saved sentiment analysis for Episode {episode_number}: {sentiment_score}, Tokens used: {token_usage}")
        total_token_count += token_usage  # Add to total token count
        return sentiment_score

# Extract episode number from episode key
def get_episode_number(episode_key):
    match = re.search(r'Episode\s*(\d+)', episode_key)
    if match:
        return int(match.group(1))
    else:
        return None

# Format character information for display
def format_characters(characters_dict):
    formatted_characters = ""
    for char_name, char_info in characters_dict.items():
        formatted_characters += f"\n- **{char_name}:**"
        for key, value in char_info.items():
            if isinstance(value, dict):
                formatted_characters += f"\n  - **{key}:**"
                for sub_key, sub_value in value.items():
                    formatted_characters += f"\n    - **{sub_key}:** {sub_value}"
            else:
                formatted_characters += f"\n  - **{key}:** {value}"
    return formatted_characters

# Format key item status for display
def format_key_item_status(key_item_status):
    return key_item_status.strip()

# Load and process episode summaries
def load_summaries(summaries_path):
    with open(summaries_path, 'r', encoding='utf-8') as f:
        summaries = json.load(f, object_pairs_hook=OrderedDict)

    documents = []
    episodes = []
    episode_keys = []
    for episode_key, content in summaries.items():
        what_if = content.get('whatIf', '')
        characters = content.get('characters', {})
        key_item_status = content.get('key_item_status', '')

        characters_str = format_characters(characters)
        key_item_status_str = format_key_item_status(key_item_status)

        page_content = f"**WhatIf:**\n{what_if}\n\n**Characters:**\n{characters_str}\n\n**Key Item Status:**\n{key_item_status_str}"

        doc = Document(
            page_content=page_content,
            metadata={
                "episode_key": episode_key,
                "key_item_status": key_item_status
            }
        )
        documents.append(doc)
        episodes.append(page_content)
        episode_keys.append(episode_key)

    return documents, episodes, episode_keys

# Load full episode contents
def load_full_contents(full_contents_path, storyline_key='Storyline 2'):
    with open(full_contents_path, 'r', encoding='utf-8') as f:
        full_contents = json.load(f, object_pairs_hook=OrderedDict)

    episodes_data = full_contents.get(storyline_key, {})

    documents = []
    episode_key_to_doc = {}
    episode_number_to_key = {}

    for episode_key, content in episodes_data.items():
        initial_records = "\n".join(content.get('initialRecords', []))
        doc = Document(page_content=initial_records, metadata={"episode_key": episode_key})
        documents.append(doc)
        episode_key_to_doc[episode_key] = doc
        episode_number = get_episode_number(episode_key)
        if episode_number is not None:
            episode_number_to_key[episode_number] = episode_key

    return documents, episode_key_to_doc, episode_number_to_key

# Create vector store for semantic search
def create_vectorstore(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(openai_api_key=client.api_key)
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

# Parse evaluation response from GPT
def parse_evaluation_response(response_text):
    score_match = re.search(r"Score\s*[:：-]?\s*(\d)", response_text, re.IGNORECASE)
    if score_match:
        score = int(score_match.group(1))
    else:
        score = None

    evaluation_reasoning = response_text.strip()

    return {
        "score": score,
        "evaluation_reasoning": evaluation_reasoning
    }

# Analyze sentiment using OpenAI API
def analyze_sentiment_openai(text):
    try:
        response = client.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a sentiment analysis tool that provides only a numerical sentiment score between 0 (very negative) and 1 (very positive)."},
                {"role": "user", "content": f"Analyze the sentiment of the following text and provide a score between 0 (very negative) and 1 (very positive):\n\n{text}\n\nSentiment score:"}
            ],
            temperature=0
        )
        sentiment_text = response.choices[0].message.content.strip()

        # Extract sentiment score
        match = re.search(r"([0-1](?:\.\d+)?)", sentiment_text)
        sentiment_score = float(match.group(1)) if match else 0.5

        # Extract token usage information
        token_usage = response['usage']['total_tokens']
        print(f"Tokens used in this request: {token_usage}")
        return sentiment_score, token_usage
    except Exception as e:
        print(f"Error during sentiment analysis: {e}")
        return 0.5, 0

# Evaluate a specific episode
def evaluate_episode(episode_number, episode_key, summaries_vectorstore, episode_key_to_doc, summaries_data, episodes, episode_keys, sentiment_results):
    llm = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=client.api_key)

    # Filter episodes up to the current episode
    filtered_episodes = []
    filtered_episode_keys = []
    for ep_num, ep_key in zip([get_episode_number(k) for k in episode_keys], episode_keys):
        if ep_num is not None and ep_num <= episode_number:
            filtered_episodes.append(episodes[episode_keys.index(ep_key)])
            filtered_episode_keys.append(ep_key)

    # Calculate similarity between episodes
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(filtered_episodes)

    current_index = filtered_episode_keys.index(episode_key)
    episode_vector = tfidf_matrix[current_index]
    cosine_similarities = cosine_similarity(episode_vector, tfidf_matrix)[0]

    # Get top similar episodes
    n = 15
    similar_indices = cosine_similarities.argsort()[-(n+1):][::-1]
    similar_indices = [i for i in similar_indices if i != current_index][:n]

    print(f"Episodes most similar to Episode {episode_number} ({episode_key}):")
    for idx in similar_indices:
        sim_episode_key = filtered_episode_keys[idx]
        sim_episode_number = get_episode_number(sim_episode_key)
        print(f"Episode {sim_episode_number} ({sim_episode_key}) - Similarity: {cosine_similarities[idx]:.2f}")

    # Perform sentiment analysis on similar episodes
    print("\nPerforming sentiment analysis...")
    sentiment_scores = []
    for idx, text in enumerate(filtered_episodes):
        ep_number = get_episode_number(filtered_episode_keys[idx])
        if ep_number is not None:
            score = analyze_or_get_sentiment(ep_number, text, sentiment_results)
            sentiment_scores.append(score)
            print(f"Episode {filtered_episode_keys[idx]} Sentiment Score: {score}")
        else:
            sentiment_scores.append(0.5)
            print(f"Episode {filtered_episode_keys[idx]} Sentiment Score: 0.5 (Default due to missing episode number)")

    episode_sentiment = sentiment_scores[current_index]
    print(f"\nSentiment analysis result for Episode {episode_number} ({episode_key}): {episode_sentiment}")

    # Select episodes based on sentiment similarity
    selected_episodes = []
    threshold = 0.4
    for idx in similar_indices:
        sim_score = sentiment_scores[idx]
        if abs(episode_sentiment - sim_score) < threshold:
            sim_episode_key = filtered_episode_keys[idx]
            sim_episode_number = get_episode_number(sim_episode_key)
            sim_similarity = cosine_similarities[idx]
            selected_episodes.append({
                'episode_number': sim_episode_number,
                'episode_key': sim_episode_key,
                'similarity': sim_similarity,
                'sentiment_score': sim_score
            })

    # Sort selected episodes by episode number
    selected_episodes.sort(key=lambda x: x['episode_number'] if x['episode_number'] is not None else float('inf'))

    print("\nFinal selected episodes (sorted by episode number):")
    for ep in selected_episodes:
        print(f"Episode {ep['episode_number']} ({ep['episode_key']}) - Similarity: {ep['similarity']:.2f}, Sentiment Score: {ep['sentiment_score']}")

    # Prepare previous episode summaries
    previous_summaries = ''
    current_episode_number = episode_number

    for ep in selected_episodes:
        sim_episode_number = ep['episode_number']
        sim_episode_key = ep['episode_key']
        sim_similarity = ep['similarity']
        sim_sentiment = ep['sentiment_score']
        summary_data = summaries_data.get((sim_episode_number, sim_episode_key), {})
        content = summary_data.get('content', '')
        previous_summaries += f"**Episode {sim_episode_number} ({sim_episode_key}) - Similarity: {sim_similarity:.2f}, Sentiment Score: {sim_sentiment}:**\n{content}\n\n"

    if not previous_summaries.strip():
        previous_summaries = "No similar episodes found."

    # Get current episode content
    current_episode_doc = episode_key_to_doc.get(episode_key)
    if current_episode_doc:
        current_episode_text = current_episode_doc.page_content
    else:
        print(f"Warning: {episode_key} not found in episodes data.")
        current_episode_text = ''

    if not current_episode_text.strip():
        print(f"Error: Current episode content for {episode_key} is empty.")
        return {
            "score": None,
            "evaluation_reasoning": f"Error: Current episode content for {episode_key} is empty."
        }

    # Get key item status
    key_item_status = summaries_data.get((episode_number, episode_key), {}).get('key_item_status', 'No key item status.')
    key_item_status_str = format_key_item_status(key_item_status)

    # Prepare evaluation prompt
    prompt = f"""
You are a meticulous literary critic specializing in narrative coherence.

As you read, pay special attention to the continuity and consistency of key items and their statuses.

**Similar Episodes Summaries:**
{previous_summaries}

**Current Episode ({episode_number}) Full Content:**
{current_episode_text}

**Current Episode ({episode_number}) Key Item Status:**
{key_item_status_str}

Please provide a critical evaluation of the current episode, focusing on:

1. **Character Consistency** - Evaluate whether the actions and dialogues of main characters in this scene align with their established traits. Note any inconsistencies and assess if they are justified by new developments.

2. **Plot Progression** - Analyze how this scene contributes to the overall story. Assess whether newly introduced elements logically extend the plot and effectively advance or resolve narrative threads.

3. **Emotional and Psychological Realism** - Review the authenticity of the main characters' emotional and psychological responses. Evaluate whether these reactions are believable and consistent with their character development and the situation.

4. **Foreshadowing and Setup for the Next Episode** - Examine how this scene prepares for subsequent developments. Consider whether it hints at future twists or sets the groundwork for upcoming narrative shifts.

5. **Continuity and Consistency in Story Elements** - Examine the episode for any inconsistencies or continuity errors, such as objects appearing or disappearing without explanation, conflicting information, or events that contradict prior established facts. Pay particular attention to items that were lost or destroyed in previous episodes but reappear without explanation. Assess how these issues impact the narrative coherence.

Provide a balanced and critical evaluation, pointing out both strengths and weaknesses. Ensure that your reasoning clearly supports the score you assign.

**Score (0-5) and Justification:**

Scoring Guidelines:
- **5 (Excellent):** The episode is highly coherent, with consistent characters, strong plot development, realistic emotions, and no major errors.

- **4 (Good):** The episode is mostly coherent, but there are small issues with characters, plot, or minor inconsistencies.

- **3 (Fair):** The episode has some noticeable inconsistencies that affect the flow.

- **2 (Poor):** The episode has major issues with consistency and continuity.

- **1 (Very Poor):** The episode is incoherent, with severe flaws and continuity problems.

- **0 (Unacceptable):** The episode is completely incoherent, with critical errors that make it nonsensical, and Error in the status of key items.
"""

    print("\nFull GPT Prompt:")
    print(prompt)
    print("-" * 50)

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        response_text = response.content
    except Exception as e:
        print(f"Error during GPT evaluation: {e}")
        return {
            "score": None,
            "evaluation_reasoning": f"Error during GPT evaluation: {e}"
        }

    evaluation_result = parse_evaluation_response(response_text)

    print(f"Score: {evaluation_result['score']}")
    print("Evaluation Reasoning:")
    print(evaluation_result['evaluation_reasoning'])

    return evaluation_result

# Analyze complex questions about the storyline
def analyze_complex_question(characters, episodes, summaries_data, episode_key_to_doc, episode_number_to_key, question_text, summaries_vectorstore, k=10):
    # Extract episode numbers mentioned in the question
    episode_numbers = re.findall(r'\b(?:episode)?\s*(\d+)\b', question_text.lower())
    specified_episodes = set(int(num) for num in episode_numbers) if episode_numbers else set()

    max_episode = max(specified_episodes) if specified_episodes else float('inf')

    # Perform similarity search on the question
    docs = summaries_vectorstore.similarity_search(question_text, k=k)
    relevant_summaries = []
    retrieved_episode_numbers = set()

    print("\nRetrieved relevant episodes and their content:")
    for doc in docs:
        episode_key = doc.metadata.get('episode_key', 'Unknown Episode')
        episode_number = get_episode_number(episode_key)
        if episode_number is not None and episode_number <= max_episode:
            relevant_summaries.append((episode_number, episode_key, doc.page_content))
            retrieved_episode_numbers.add(episode_number)

    # Add specified episodes if missing from search results
    for ep_num in specified_episodes:
        if ep_num not in retrieved_episode_numbers:
            episode_key = episode_number_to_key.get(ep_num)
            if episode_key:
                doc = episode_key_to_doc.get(episode_key)
                if doc:
                    relevant_summaries.append((ep_num, episode_key, doc.page_content))
                    print(f"Manually added Episode {ep_num} ({episode_key})")

    # Sort summaries by episode number
    relevant_summaries.sort(key=lambda x: x[0])

    formatted_summaries = ""
    for episode_number, episode_key, content in relevant_summaries:
        print(f"Episode {episode_number} ({episode_key})")
        formatted_summaries += f"**Episode {episode_number} ({episode_key}):**\n{content}\n\n"

    # Prepare analysis prompt
    prompt = f"""
You are an expert in literary analysis.

The user has the following question about events up to and including Episode {max_episode}:

"{question_text}"

Based on the following episode summaries (which only include information up to Episode {max_episode}), please provide a detailed analysis to answer the user's question. Focus on character motivations, emotional changes, and plot development across the relevant episodes. Do not consider or mention any events that occur after Episode {max_episode}.

**Relevant Episode Summaries:**
{formatted_summaries}

Provide a clear and concise response that directly addresses the user's inquiry, ensuring you only discuss events and character development up to Episode {max_episode}.
"""

    llm = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=client.api_key)

    print("\nFull GPT Prompt for Complex Question:")
    print(prompt)
    print("-" * 50)

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content
    except Exception as e:
        print(f"Error during complex analysis: {e}")
        return None

# Initialize data for analysis
def initialize_data(summaries_path, full_contents_path, SENTIMENT_RESULTS_PATH):
    summaries_documents, episodes, episode_keys = load_summaries(summaries_path)
    summaries_vectorstore = create_vectorstore(summaries_documents)
    full_contents_documents, episode_key_to_doc, episode_number_to_key = load_full_contents(full_contents_path)
    sentiment_results = load_sentiment_results(SENTIMENT_RESULTS_PATH)

    summaries_data = {}
    for doc in summaries_documents:
        episode_key = doc.metadata.get('episode_key', '')
        key_item_status = doc.metadata.get('key_item_status', '')
        episode_number = get_episode_number(episode_key)
        if episode_number is not None:
            summaries_data[(episode_number, episode_key)] = {
                'content': doc.page_content,
                'key_item_status': key_item_status
            }

    return summaries_vectorstore, episodes, episode_keys, episode_key_to_doc, episode_number_to_key, summaries_data, sentiment_results

# Classify user input as evaluation or analysis request
def classify_user_input(user_input):
    llm = ChatOpenAI(model_name="gpt-4o", openai_api_key=client.api_key)
    prompt = f"""
The user has input the following question or request:

"{user_input}"

Please determine whether the user wants to:

1. **Evaluate an episode**
2. **Analyze characters or plot**

Respond only with: "Evaluate Episode" or "Analyze Characters/Plot".
"""
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()
    except Exception as e:
        print(f"Error during classification: {e}")
        return None

# Handle episode evaluation requests
def handle_evaluation_request(user_input, episode_number_to_key, summaries_vectorstore, episode_key_to_doc,
                              summaries_data, episodes, episode_keys, sentiment_results, evaluation_results):
    episode_number_matches = re.findall(r'\b(?:episode)?\s*(\d+)\b', user_input.lower())
    if episode_number_matches:
        episode_number = int(episode_number_matches[0])
        episode_key = episode_number_to_key.get(episode_number)

        if episode_key:
            print(f"Evaluating {episode_key}...")
            evaluation_result = evaluate_episode(
                episode_number, episode_key, summaries_vectorstore,
                episode_key_to_doc, summaries_data, episodes, episode_keys, sentiment_results
            )
            if evaluation_result:
                print(f"Score: {evaluation_result['score']}")
                print("Evaluation Reasoning:")
                print(evaluation_result['evaluation_reasoning'])
                evaluation_results[episode_key] = evaluation_result
        else:
            print(f"Episode {episode_number} does not exist.")
    else:
        print("Could not extract episode number from input.")

# Handle analysis requests
def handle_analysis_request(user_input, summaries_data, episode_key_to_doc,
                            episode_number_to_key, summaries_vectorstore):
    result = analyze_complex_question([], [], summaries_data,
                                      episode_key_to_doc, episode_number_to_key,
                                      user_input, summaries_vectorstore, k=15)
    if result:
        print(f"\nAnalysis result:\n{result}\n")
    else:
        print("Failed to generate analysis result.")

# Main interactive loop
def interactive_loop(summaries_vectorstore, episodes, episode_keys, episode_key_to_doc,
                     episode_number_to_key, summaries_data, sentiment_results):
    evaluation_results = OrderedDict()

    while True:
        user_input = input("Please enter your question (type 'exit' to quit): ")
        if user_input.lower().strip() == 'exit':
            print("Program exited.")
            break

        classification = classify_user_input(user_input)

        if classification == "Evaluate Episode":
            handle_evaluation_request(user_input, episode_number_to_key, summaries_vectorstore,
                                      episode_key_to_doc, summaries_data, episodes, episode_keys,
                                      sentiment_results, evaluation_results)
        elif classification == "Analyze Characters/Plot":
            handle_analysis_request(user_input, summaries_data,
                                    episode_key_to_doc, episode_number_to_key, summaries_vectorstore)
        else:
            print("Unrecognized classification.")

    return evaluation_results

# Main function
def main():
    summaries_path = './data/stoeyline26 summary_key_item.json'
    full_contents_path = './data/Storyline_26.json'
    output_path = './data/evaluation_results.json'
    SENTIMENT_RESULTS_PATH = './data/save_score_episode26.json'
    total_token_count = 0
    
    # Initialize data
    summaries_vectorstore, episodes, episode_keys, episode_key_to_doc, episode_number_to_key, summaries_data, sentiment_results = initialize_data(
        summaries_path, full_contents_path, SENTIMENT_RESULTS_PATH)

    # Start interactive loop
    evaluation_results = interactive_loop(
        summaries_vectorstore, episodes, episode_keys,
        episode_key_to_doc, episode_number_to_key,
        summaries_data, sentiment_results
    )

    # Save results on exit
    save_sentiment_results(sentiment_results, SENTIMENT_RESULTS_PATH)
    print(f"Total tokens consumed: {total_token_count}")

    if evaluation_results:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
        print(f"All evaluation results have been saved to {output_path}")


if __name__ == "__main__":
    main()