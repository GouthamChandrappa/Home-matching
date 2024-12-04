# HomeMatch: AI-Powered Real Estate Matching Application

## Project Overview

HomeMatch is an innovative real estate application that leverages AI to generate, match, and personalize property listings based on buyer preferences. The application uses OpenAI's language models and ChromaDB for semantic search to create a personalized home-finding experience.

## Features

- Synthetic Real Estate Listing Generation
- Buyer Preference Collection
- Semantic Matching of Listings
- Personalized Listing Descriptions
- Vector Database Integration

## Prerequisites

- Python 3.10+
- OpenAI API Key
- Anaconda or Virtualenv (recommended)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/GouthamChandrappa/Home-matching.git
cd Home-matching
```

Create a virtual environment:
```
conda create -n homematch python=3.10
conda activate homematch
```
Install dependencies:
```
pip install -r requirements.txt
```
Set up OpenAI API Key:
```
export OPENAI_API_KEY='your_openai_api_key_here'
```
Configuration
Modify openai.api_key in the script with your actual OpenAI API key
Adjust the output directory path in store_listings() method if needed

Running the Application:
```
python homematch.py
```
The application will:
-Generate synthetic real estate listings
-Store listings in a text file
-Interactively collect your home preferences
-Find and personalize matching listings

Dependencies:
```
-chromadb==0.3.21
-pydantic==1.10.11
-openai
-sentence-transformers
-numpy
-pandas
```

How It Works:
-Listing Generation: Uses OpenAI to create realistic property listings
-Preference Collection: Interactive questionnaire to understand buyer needs
-Semantic Matching: Uses vector embeddings to find the most relevant listings
-Personalization: Tailors listing descriptions to individual preferences

Potential Improvements:
-Add more sophisticated filtering
-Implement persistent storage
-Enhance AI-driven matching algorithms

Troubleshooting:
-Ensure OpenAI API key is valid
-Check internet connectivity
-Verify all dependencies are installed



