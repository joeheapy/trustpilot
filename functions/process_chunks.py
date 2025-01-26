import pandas as pd
import os
from openai import OpenAI
from datetime import datetime
import json
from openai import AsyncOpenAI
from dotenv import load_dotenv
from functions.clean_markdown import clean_markdown

# This function processes each chunk file and sends the reviews to the OpenAI API for analysis. The function returns the sentiment analysis for each review.

# Load environment variables
load_dotenv()

SYSTEM_PROMPT = """You are a data processing assistant. You are tasked with analyzing the sentiment of customer reviews for a company. The reviews are in the form of text data. Your task is to read each review and determine the sentiment.You should then provide a brief summary of the sentiment analysis for each review - sentiment summary. The reviews are from a variety of sources, so you may encounter different writing styles and topics. Your goal is to provide an accurate and consistent analysis of the sentiment of each review. Please highlight specific details that evidence the customer experience.

Date format always "YYYY-MM-DD"

Return ONLY this exact JSON structure:

{
  "date": "string",
  "title": "string",
  "rating": "integer",
  "sentimentSummary": "string"
}
"""


# Initialize OpenAI client
client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))


# This function processes each chunk file and sends the reviews to the OpenAI API for analysis. The function returns the sentiment analysis for each review.
async def process_chunks():
    """Process each chunk file and send to OpenAI API"""
    chunk_files = sorted([f for f in os.listdir('data-chunks') if f.endswith('.json')])
    
    for chunk_file in chunk_files:
        try:
            # Read JSON chunk file
            chunk_path = os.path.join('data-chunks', chunk_file)
            with open(chunk_path, 'r') as f:
                chunk_data = json.load(f)
            chunk_df = pd.DataFrame(chunk_data)
            
            # Format reviews
            reviews = []
            for _, row in chunk_df.iterrows():
                review = (
                    f"Date: {row['reviewDateOfExperience']}\n"
                    f"Title: {row['reviewTitle']}\n"
                    f"Rating: {row['reviewRatingScore']}/5\n"
                    f"Description: {row['reviewDescription']}\n"
                    f"{'='*50}"
                )
                reviews.append(review)
            
            chunk_text = "\n".join(reviews)
            
            print(f"Sending batch to OpenAI: {chunk_file}...")

            # Wait for API response
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": chunk_text}
                ]
            )
            # Changing this now.
            if response.choices and response.choices[0].message.content:
                # Clean markdown before saving
                cleaned_content = clean_markdown(response.choices[0].message.content)
                
                analyzed_dir = "analyzed-chunks"
                if not os.path.exists(analyzed_dir):
                    os.makedirs(analyzed_dir)
                    
                output_file = os.path.join(analyzed_dir, f"analyzed_{chunk_file}")
                with open(output_file, 'w') as f:
                    json.dump({
                        "response": cleaned_content,
                    }, f, indent=2)
            
        except Exception as e:
            print(f"Error processing {chunk_file}: {str(e)}")
            continue