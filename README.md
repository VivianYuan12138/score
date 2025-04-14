# SCORE: Story Coherence and Retrieval Enhancement

![SCORE Architecture](./score_framework.png)

> **Qiang Yi\*, Yangfan He\*, Jianhui Wang\*, Xinyuan Song, Shiyao Qian, Miao Zhang, Li Sun, Tianyu Shi.**  
> *SCORE: Story Coherence and Retrieval Enhancement for AI Narratives.*  
> arXiv preprint arXiv:2503.23512, 2025.  
> [[arXiv Link]](https://arxiv.org/abs/2503.23512)

## Overview

SCORE is a lightweight LLM-based framework designed to evaluate and enhance story coherence in AI-generated narratives. The framework addresses three critical aspects of narrative consistency:

1. **Dynamic State Tracking**  
   Monitors and maintains consistency in the states of key narrative elements across episodes.

2. **Context-Aware Summarization**  
   Generates structured episode summaries that capture character development, relationships, and emotional arcs.

3. **Hybrid Retrieval**  
   Implements a RAG-based approach combining semantic similarity and sentiment analysis for coherent multi-episode reasoning.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python step1.py  # Initial processing
python step2.py  # Episode analysis
python step3.py  # Key items analysis
python step4.py  # Interactive evaluation
```

## Core Components

### Data Processing Pipeline

- **Step 1: Initial Processing**  
  - Text splitting and embedding generation
  - Vector store creation for semantic search

- **Step 2: Episode Analysis**  
  - Character and plot analysis
  - Key item tracking
  - Structured summary generation

- **Step 3: Key Items Analysis**  
  - Cross-episode item tracking
  - Importance analysis
  - State consistency verification

- **Step 4: Evaluation System**  
  - Interactive analysis interface
  - Quality evaluation
  - Sentiment analysis
  - Complex question answering

### Data Structure

- **Input**: 
  - `Storyline_26.json` (Main narrative data)
  - `stoeyline26每集的key_item.json` (Key items data)

- **Output**:
  - `episodes27_analysis.json` (Episode analysis)
  - `stoeyline26 summary_key_item.json` (Key items summary)
  - `evaluation_results.json` (Evaluation results)
  - `save_score_episode26.json` (Sentiment scores)

## Configuration

Configure `config.py` with:
- OpenAI API key


## Dependencies

```bash
openai>=1.12.0
langchain>=0.1.0
langchain-openai>=0.0.5
langchain-community>=0.0.10
scikit-learn>=1.3.0
numpy>=1.24.0
pandas>=2.0.0
```

## Citation

```bibtex
@article{yi2025score,
  title={SCORE: Story Coherence and Retrieval Enhancement for AI Narratives},
  author={Yi, Qiang and He, Yangfan and Wang, Jianhui and Song, Xinyuan and Qian, Shiyao and Zhang, Miao and Sun, Li and Shi, Tianyu},
  journal={arXiv preprint arXiv:2503.23512},
  year={2025}
}
``` 