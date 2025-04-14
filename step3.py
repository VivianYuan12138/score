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

def merge_key_items_into_episodes(episodes_json_path, key_items_json_path, output_json_path):
    """
    Merge key item statuses into the existing episode JSON data,
    and insert 'key_item_status' between 'whatIf' and 'characters'.

    Parameters:
        episodes_json_path (str): Path to the JSON file containing episode plot and character info.
        key_items_json_path (str): Path to the JSON file containing per-episode key item statuses.
        output_json_path (str): Path to save the merged output JSON data.
    """
    try:
        # Load episode data
        with open(episodes_json_path, 'r', encoding='utf-8') as f:
            episodes_data = json.load(f, object_pairs_hook=OrderedDict)
        print(f"Successfully loaded episode data from {episodes_json_path}")
    except Exception as e:
        print(f"Error reading episode JSON file: {str(e)}")
        return

    try:
        # Load key item data
        with open(key_items_json_path, 'r', encoding='utf-8') as f:
            key_items_data = json.load(f)
        print(f"Successfully loaded key item data from {key_items_json_path}")
    except Exception as e:
        print(f"Error reading key item JSON file: {str(e)}")
        return

    # Iterate through each episode and add key item status to the corresponding episode
    for episode_key in episodes_data.keys():
        # Extract episode number using regex
        match = re.match(r'^Episode\s+(\d+)$', episode_key)
        if not match:
            print(f"Skipping unmatched episode key: {episode_key}")
            continue
        episode_number = int(match.group(1))

        # Get the key item status for the corresponding episode
        key_items_text = key_items_data.get(str(episode_number), "")

        # Get current episode data
        current_episode_data = episodes_data[episode_key]

        # Create a new OrderedDict to maintain key order
        merged_episode = OrderedDict()

        # Insert 'key_item_status' between 'whatIf' and 'characters'
        for key, value in current_episode_data.items():
            merged_episode[key] = value
            if key.lower() == 'whatif':
                merged_episode['key_item_status'] = key_items_text

        # Update the episode data
        episodes_data[episode_key] = merged_episode

    # Write the updated data to a new JSON file
    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(episodes_data, f, indent=2, ensure_ascii=False)
        print(f"Successfully merged data and saved to {output_json_path}")
    except Exception as e:
        print(f"Error writing merged JSON file: {str(e)}")

if __name__ == "__main__":
    # Define file paths
    episodes_json_path = './data/episodes27_analysis.json'  # Replace with your episode data JSON file path
    key_items_json_path = './data/stoeyline26每集的key_item.json'  # Replace with your key item data JSON file path
    output_json_path = './data/stoeyline26 summary_key_item.json'  # Define output path for merged data

    # Call merge function
    merge_key_items_into_episodes(episodes_json_path, key_items_json_path, output_json_path)
