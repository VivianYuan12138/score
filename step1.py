import os
import json
import re
from collections import OrderedDict, defaultdict
from openai import OpenAI

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from config import openai

client = OpenAI(api_key=openai.api_key)

def get_episode_number(episode_key):
    # Extract the episode number from the episode key, e.g., "Episode 1" becomes 1.
    match = re.search(r'Episode\s*(\d+)', episode_key, re.IGNORECASE)
    if match:
        return int(match.group(1))
    else:
        return None

def load_summaries(summaries_path):
    # Load summary data and create a list of Document objects.
    with open(summaries_path, 'r', encoding='utf-8') as f:
        summaries = json.load(f, object_pairs_hook=OrderedDict)

    documents = []
    for episode_key, content in summaries.items():
        page_content = content.get('whatIf', '')
        doc = Document(page_content=page_content, metadata={"episode_key": episode_key})
        documents.append(doc)

    return documents

def load_full_contents(full_contents_path, storyline_key='Storyline 3'):
    """
    Load full content data and create a list of Document objects and a mapping.
    Includes information on key_item_status.
    """
    with open(full_contents_path, 'r', encoding='utf-8') as f:
        full_contents = json.load(f, object_pairs_hook=OrderedDict)

    episodes_data = full_contents.get(storyline_key, {})

    documents = []
    episode_key_to_doc = {}

    for episode_key, content in episodes_data.items():
        initial_records = "\n".join(content.get('initialRecords', []))
        key_item_status = content.get('key_item_status', '')
        doc_content = f"{initial_records}\n\n**Key Item Status:**\n{key_item_status}"
        doc = Document(page_content=doc_content, metadata={"episode_key": episode_key})
        documents.append(doc)
        episode_key_to_doc[episode_key] = {
            "full_content": initial_records,
            "key_item_status": key_item_status
        }

    return documents, episode_key_to_doc


def create_vectorstore(documents):
    """
    Create a vector store (FAISS) for similarity retrieval.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

def categorize_feedback(feedback_list):
    """
    Categorize feedback based on type.
    """
    categorized_feedback = defaultdict(list)
    for feedback in feedback_list:
        category = feedback.get("category", "Uncategorized")
        comment = feedback.get("comment", "")
        categorized_feedback[category].append(comment)
    return categorized_feedback

def apply_feedback_with_coherence(categorized_feedback):
    """
    Generate improvement guidelines based on categorized feedback.
    """
    guidelines = {}

    # Story coherence improvements
    if "Story Coherence" in categorized_feedback:
        guidelines["coherence"] = "Ensure smooth transitions between events and clarify character motivations. Add additional context between story scenes to prevent abrupt jumps."

    # Emotion improvements
    if "Needs More Emotion" in categorized_feedback:
        guidelines["emotion"] = "Add more internal dialogue and detailed descriptions of character emotions to build emotional depth."

    # Narration improvements
    if "Better Narration" in categorized_feedback:
        guidelines["narration"] = "Improve scene pacing with more narrations between dialogues to build tension and provide a better flow between story events."

    # Character interaction improvements
    if "Character Interaction" in categorized_feedback:
        guidelines["interaction"] = "Increase the frequency and depth of interactions between main characters to strengthen their relationship and provide context for their motivations."

    # Tone improvements
    if "Tone Issues" in categorized_feedback:
        guidelines["tone"] = "Ensure consistent tone across scenes; avoid dialogues that feel out of place in serious situations."

    return guidelines

def generate_feedback_and_guidelines(evaluation_reasoning):
    """
    Generate improvement guidelines based on feedback.
    Dynamically generate feedback from evaluation reasoning.
    """
    feedback_prompt = f"""
Based on the following evaluation reasons, categories and specific suggestions for improvement are extracted.

**evalution)reasoning:**
{evaluation_reasoning}

**Please return it in the form of a dictionary, with the key being the category and the value being the specific suggestion.**
"""

    llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai.api_key)
    response_message = llm.invoke([HumanMessage(content=feedback_prompt)])
    response_text = response_message.content

    try:
        improvement_guidelines = json.loads(response_text)
    except json.JSONDecodeError:
        print("Error: Failed to parse improvement guidelines from model response.")
        improvement_guidelines = {}

    return improvement_guidelines

def parse_evaluation_response(response_text):
    # Extract score and evaluation reasoning.
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

def evaluate_episode(episode_number, episode_key, summaries_vectorstore, episode_key_to_doc, show_retrieved_summaries=False):
    """
    Evaluate the specified episode using RAG to retrieve relevant summaries and generate an evaluation.
    """
    llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai.api_key)

    # Get current episode content and key_item_status
    current_episode_data = episode_key_to_doc.get(episode_key)
    if current_episode_data:
        current_episode_text = current_episode_data["full_content"]
        key_item_status = current_episode_data["key_item_status"]
    else:
        print(f"Warning: {episode_key} not found in episodes data.")
        current_episode_text = ''
        key_item_status = ''

    if not current_episode_text.strip():
        print(f"Error: Current episode content for {episode_key} is empty.")
        return {
            "score": None,
            "evaluation_reasoning": f"Error: Current episode content for {episode_key} is empty."
        }

    # Use current episode content as query to retrieve most relevant summaries from the vector store
    query = current_episode_text
    retrieved_summaries_docs = summaries_vectorstore.similarity_search(query, k=15)  # Retrieve 10 most relevant summaries

    # Filter retrieved summaries to only keep those with episode number less than the current one
    filtered_summaries_docs = []
    for doc in retrieved_summaries_docs:
        ep_key = doc.metadata.get('episode_key', '')
        ep_number = get_episode_number(ep_key)
        if ep_number is not None and ep_number < episode_number:
            filtered_summaries_docs.append(doc)
        if len(filtered_summaries_docs) >= 15:
            break  # Keep at most 15 relevant summaries

    # Extract retrieved summaries
    previous_summaries = ''
    print("\nRetrieved Relevant Summaries:")
    for idx, doc in enumerate(filtered_summaries_docs, 1):
        ep_key = doc.metadata.get('episode_key', '')
        summary_text = doc.page_content
        previous_summaries += f"{ep_key}:\n{summary_text}\n\n"
        if show_retrieved_summaries:
            print(f"Summary {idx}: {ep_key}")
            print(summary_text)
            print("-" * 50)

    if not previous_summaries.strip():
        previous_summaries = "No relevant previous episodes."
        
    # Construct prompt for evaluation, including key_item_status
    prompt = f"""
You are a meticulous literary critic specializing in narrative coherence.

As you read, pay special attention to the continuity and consistency of key items and their statuses.

**Relevant Previous Episodes Summaries:**
{previous_summaries}

**Current Episode ({episode_number}) Full Content:**
{current_episode_text}

**Key Item Status:**
{key_item_status}

Please provide a critical evaluation of the current episode, focusing on:

1. **Character Consistency** - Evaluate whether the actions and dialogues of main characters in this scene align with their established traits. Note any inconsistencies and assess if they are justified by new developments. Give a score based on the Score (0-5) and Justification.

2. **Plot Progression** - Analyze how this scene contributes to the overall story. Assess whether newly introduced elements logically extend the plot and effectively advance or resolve narrative threads. Give a score based on the Score (0-5) and Justification.

3. **Emotional and Psychological Realism** - Review the authenticity of the main characters' emotional and psychological responses. Evaluate whether these reactions are believable and consistent with their character development and the situation. Give a score based on the Score (0-5) and Justification.

4. **Continuity and Consistency in Story Elements** - Examine the episode for any inconsistencies or continuity errors, such as objects appearing or disappearing without explanation, conflicting information, or events that contradict prior established facts. Pay particular attention to items that were lost or destroyed in previous episodes but reappear without explanation. Assess how these issues impact the narrative coherence. Give a score based on the Score (0-5) and Justification.

**Also, please check the "Key Item Status" section specifically for any consistency errors as with this episode (key_item_status should be correct). Make sure the status of each key item is consistent with the status of the current episode. Report any discrepancies or errors. Give a score based on the Score (0-5) and Justification. **

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

    # Call GPT API to perform evaluation
    response_message = llm.invoke([HumanMessage(content=prompt)])
    response_text = response_message.content

    # Parse response
    evaluation_result = parse_evaluation_response(response_text)

    # Generate improvement suggestions
    improvement_guidelines = generate_feedback_and_guidelines(evaluation_result['evaluation_reasoning'])
    evaluation_result['improvement_guidelines'] = improvement_guidelines

    # Return evaluation result
    return evaluation_result


def update_summary_and_key_items(previous_summary, previous_key_items, current_scene, episode_number, continuity_analysis):

    # Use content from initialRecords
    current_scene_text = '\n'.join(current_scene['initialRecords'])

    # Convert previous key item statuses to text
    key_items_text = ""
    for item_name, item_info in previous_key_items.items():
        key_items_text += f"- {item_name}: [Status: {item_info['status']}, Last Known Location/Owner: {item_info['location']}, Current Importance: {item_info['importance']}]\n"

    # Output the key_items_text passed to GPT
    print("\n--- key_items_text passed to GPT ---\n")
    print(key_items_text)
    print("\n--- End of key_items_text ---\n")

    prompt = f""" ... """  # [unchanged prompt omitted for brevity]

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {"role": "system", "content": "You are a professional story analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )

        if response and response.choices and response.choices[0].message:
            content = response.choices[0].message.content.strip()

            # Print updated summary and key items generated by GPT
            print("\n--- Updated summary and key items generated by GPT ---\n")
            print(content)
            print("\n--- End of GPT generated content ---\n")

            # Split updated summary and key items
            if "**Updated Key Items and Their Statuses:**" in content:
                summary_part, key_items_part = content.split("**Updated Key Items and Their Statuses:**", 1)
                updated_summary = summary_part.replace("**Updated Summary:**", "").strip()
                key_items_text = key_items_part.strip()

                # Parse key items and their statuses
                updated_key_items = {}
                items = key_items_text.split("\n\n")
                for item in items:
                    lines = item.strip().split('\n')
                    if len(lines) >= 4:
                        item_name_line = lines[0]
                        status_line = lines[1]
                        location_line = lines[2]
                        importance_line = lines[3]

                        item_name = item_name_line.replace("Item Name:", "").strip()
                        status = status_line.replace("- Current Status:", "").strip()
                        location = location_line.replace("- Last Known Location/Owner:", "").strip()
                        importance = importance_line.replace("- Current Importance:", "").strip()

                        updated_key_items[item_name] = {
                            'status': status,
                            'location': location,
                            'importance': importance
                        }
                return updated_summary, updated_key_items
            else:
                # If there's no clear split, return the whole content as summary, keep key items unchanged
                updated_summary = content
                updated_key_items = previous_key_items
                return updated_summary, updated_key_items
        else:
            return previous_summary, previous_key_items
    except Exception as e:
        print(f"Error updating summary and key items: {str(e)}")
        return previous_summary, previous_key_items

def evaluate_scene(previous_summary, previous_key_items, current_scene, next_scene, episode_number):

    # Use content from initialRecords
    current_scene_text = '\n'.join(current_scene['initialRecords'])
    next_scene_text = '\n'.join(next_scene['initialRecords']) if next_scene else "without next scene"

    # Convert previous key item statuses to text
    key_items_text = ""
    for item_name, item_info in previous_key_items.items():
        key_items_text += f"- {item_name}: [Status: {item_info['status']}, Last Known Location/Owner: {item_info['location']}, Current Importance: {item_info['importance']}]\n"

    prompt = f""" ... """  # [unchanged prompt omitted for brevity]

    # Print evaluation prompt sent to GPT
    print("\n--- Evaluation prompt sent to GPT ---\n")
    print(prompt)
    print("\n--- End of evaluation prompt ---\n")

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {"role": "system", "content": "You are a meticulous literary critic specializing in narrative coherence."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        if response and response.choices and response.choices[0].message:
            content = response.choices[0].message.content.strip()

            # Print GPT's evaluation response
            print("\n--- GPT evaluation response ---\n")
            print(content)
            print("\n--- End of GPT evaluation response ---\n")

            # Extract score
            match = re.search(r"Score\s*[:：]\s*(\d+)", content, re.IGNORECASE)
            score = int(match.group(1)) if match else None

            # Extract analysis of point 5
            continuity_analysis = ""
            pattern = r"5\.\s*\*\*Continuity and Consistency in Story Elements\*\*([\s\S]*?)(?=\n\d|$)"
            match = re.search(pattern, content)
            if match:
                continuity_analysis = match.group(1).strip()

            reasoning = content
            return score, reasoning, continuity_analysis
        else:
            return None, None, None
    except Exception as e:
        print(f"Error evaluating episode: {str(e)}")
        return None, None, None


def process_storyline_for_key_items(data):
    """
    Processes a storyline dict, evaluates continuity and updates summary/key item states.
    Returns the updated storyline data with key_items_status added.
    """
    scenes = []
    for storyline_key, storyline_value in data.items():
        for episode_key, episode_value in storyline_value.items():
            match = re.match(r'^Episode\s+(\d+)$', episode_key)
            if not match:
                continue
            episode_number = int(match.group(1))
            scenes.append({
                'storyline_key': storyline_key,
                'episode_key': episode_key,
                'episode_number': episode_number,
                'initialRecords': episode_value.get('initialRecords', [])
            })

    scenes.sort(key=lambda x: x['episode_number'])

    previous_summary = ""
    previous_key_items = {}

    for i, scene in enumerate(scenes):
        storyline_key = scene['storyline_key']
        episode_key = scene['episode_key']
        episode_number = scene['episode_number']
        next_scene = scenes[i + 1] if i + 1 < len(scenes) else None

        print(f"Processing {episode_key}")
        continuity_analysis = evaluate_scene(previous_summary, previous_key_items, scene, next_scene, episode_number)
        updated_summary, updated_key_items = update_summary_and_key_items(previous_summary, previous_key_items, scene, episode_number, continuity_analysis)

        # Debug output
        print(f"Key items status for {episode_key}:")
        for item_name, item_info in updated_key_items.items():
            print(f"Item Name: {item_name}")
            print(f"- Current Status: {item_info['status']}")
            print(f"- Last Known Location/Owner: {item_info['location']}")
            print(f"- Current Importance: {item_info['importance']}\n")

        previous_summary = updated_summary
        previous_key_items = updated_key_items

        data[storyline_key][episode_key]['key_items_status'] = updated_key_items
        for field in ['initialRecords', 'score', 'evaluation_reasoning', 'whatIf', 'characters']:
            data[storyline_key][episode_key].pop(field, None)

    return data


if __name__ == "__main__":
    try:
        with open('./data/Storyline_26.json', 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        updated_data = process_storyline_for_key_items(raw_data)
        with open('./data/stoeyline26每集的key_item.json', 'w', encoding='utf-8') as f:
            json.dump(updated_data, f, indent=2, ensure_ascii=False)
        print("Saved to stoeyline26每集的key_item.json")
    except Exception as e:
        print(f"Error: {e}")

