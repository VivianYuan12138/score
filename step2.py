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

def load_data(file_path):
    """Load JSON data file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except Exception as e:
        print(f"Error loading data from {file_path}: {str(e)}")
        return None

def save_data(data, file_path):
    """Save JSON data to file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
        print(f"Data successfully saved to {file_path}")
    except Exception as e:
        print(f"Error saving data to {file_path}: {str(e)}")

def extract_json_from_text(text):
    """Extract JSON string from text and parse it as a dictionary."""
    try:
        json_str = text.strip()
        # Find the position of the first and last curly braces
        start = json_str.find('{')
        end = json_str.rfind('}') + 1
        json_str = json_str[start:end]
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {str(e)}")
        return None

def analyze_episode(episode_name, episode_data, important_items):
    """Call OpenAI API to analyze a single episode."""
    prompt = f"""
    You are a professional story analyst. Please analyze the following episode data and provide detailed analysis for each character.

    **Please note: All output should be in English, including character names and field names.**

    **Please strictly follow these requirements:**

    - **Output only JSON format data, do not add any additional text, explanations, or comments.**
    - **Ensure the output JSON format is correct and can be parsed by a JSON parser.**
    - **Only include items from the provided `important_items` list. Do not add or identify other items.**

    **For each episode, please output according to the following structure:**

    {{
      "{episode_name}": {{
        "whatIf": "{episode_data.get('whatIf', '')}",
        "characters": {{
          "Character Name": {{
            "Interactions_with_Key_Items": {{
              "Item Name": "Description of interaction with the item [Status]"
            }},
            "Actions": "Overall description of character's actions",
            "Relationships": {{
              "Relationship with other character": "Description of relationship"
            }},
            "Emotions": {{
              "Emotion name": "Description of emotion"
            }}
          }},
          ...
        }}
      }}
    }}

    **Here is the episode data:**

    {json.dumps(episode_data['initialRecords'], ensure_ascii=False, indent=2)}

    **Here are the important items:**

    {json.dumps(important_items, ensure_ascii=False, indent=2)}

    **How to generate the Status field:**

    - Based on the events in `initialRecords`, determine the current status of the item. For example:
      - If an item is lost, `[Lost on cliff]`.
      - If an item is hidden, `[Hidden by principal]`.
      - For other necessary statuses, please judge based on the plot and note in the description.

    **Please ensure:**

    - **Output only JSON format data, do not add any additional text.**
    - **All text should be in English.**
    - **JSON format is strictly correct and can be parsed.**
    - **Only include interactions with items from the `important_items` list.**
    - **In `Interactions_with_Key_Items`, the item's status should be included in the description, such as `[Status]`.**
    - **If there are no interactions with important items, keep the field empty.**
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a professional story analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )

        # Get response content
        content = response.choices[0].message.content.strip()
        print(f"Response Content for {episode_name}:")
        print(content)

        # Extract JSON from response
        analysis = extract_json_from_text(content)
        if analysis:
            return analysis
        else:
            print(f"Failed to parse JSON for {episode_name}.")
            return None

    except Exception as e:
        print(f"Error analyzing {episode_name}: {str(e)}")
        return None

def analyze_storyline_episodes(story_data, important_items):
    """
    Analyze each episode in the storyline and return structured results.
    Parameters:
        story_data (dict): storyline JSON loaded from file.
        important_items (list): list of important item names.
    Returns:
        dict: structured analysis result per episode.
    """
    analysis_results = {}
    for storyline, episodes in story_data.items():
        for episode_name, episode_data in episodes.items():
            print(f"Analyzing {episode_name}...")
            analysis = analyze_episode(episode_name, episode_data, important_items)
            if analysis:
                analysis_results.update(analysis)
            else:
                print(f"Failed to analyze {episode_name}.")
    return analysis_results


if __name__ == "__main__":
    story_data = load_data('./data/Storyline_26.json')
    important_items_data = load_data('./data/stoeyline26每集的key_item.json')

    if not story_data or not important_items_data:
        print("❌ Failed to load input data.")
    else:
        important_items = important_items_data.get('important_items', [])
        results = analyze_storyline_episodes(story_data, important_items)
        save_data(results, './data/episodes27_analysis.json')
        print("✅ Episode analysis complete.")