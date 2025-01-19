import pandas as pd
import os
from openai import OpenAI
from datetime import datetime
import json
import shutil
import asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

# Set chunk size
NUM_REVIEWS_PER_CHUNK = 25
# Delay between API calls (in seconds) 
NUM_CHUNKS = 2


SYSTEM_PROMPT = "You are a data processing assistant. You are tasked with analyzing the sentiment of customer reviews for a company. The reviews are in the form of text data. Your task is to read each review and determine the sentiment of the review (positive, negative, or neutral). You should then provide a brief summary of the sentiment analysis for each review. The reviews are from a variety of sources, so you may encounter different writing styles and topics. Your goal is to provide an accurate and consistent analysis of the sentiment of each review. Please highlight specific details that evidence the customer's experience."


def clean_markdown(text):
    """Remove markdown formatting characters from text"""
    import re
    patterns = [
        (r'\*\*(.*?)\*\*', r'\1'),  # Remove bold
        (r'#{1,6}\s*', ''),         # Remove headers
        (r'\[(.*?)\]\(.*?\)', r'\1'), # Remove links
        (r'[*_]{1,2}(.*?)[*_]{1,2}', r'\1'), # Remove emphasis
        (r'`{1,3}.*?`{1,3}', ''),   # Remove code blocks
        (r'\n{3,}', '\n\n')         # Clean multiple newlines
    ]
    
    cleaned = text
    for pattern, replacement in patterns:
        cleaned = re.sub(pattern, replacement, cleaned)
    return cleaned.strip()

def get_input_file():
    """Get the input file path from raw_trustpilot_data directory"""
    # Define data directory
    data_dir = "raw_trustpilot_data"
    
    try:
        # Check if directory exists
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Directory not found: {data_dir}")
        
        # Get JSON files from directory
        json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
        
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {data_dir}")
        
        # Get first JSON file (assuming single file)
        input_file = os.path.join(data_dir, json_files[0])
        print(f"Found input file: {input_file}")
        return input_file
        
    except Exception as e:
        print(f"Error finding input file: {str(e)}")
        raise

# Replace hard-coded path with function call
input_file = get_input_file()

# Get base name from input file
base_name = os.path.splitext(os.path.basename(input_file))[0]

# Initialize OpenAI client
client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# This deletes and recreates the working directories to ensure a clean start
def initialize_directories():
    """Initialize working directories by removing and recreating them"""
    directories = [
        'analyzed-chunks',
        'data-chunks',
        'summarized-reviews'
    ]
    
    print(f"\nInitializing directories at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    for directory in directories:
        try:
            # Remove directory and contents if exists
            if os.path.exists(directory):
                shutil.rmtree(directory)
                print(f"Removed existing directory: {directory}")
            
            # Create new empty directory
            os.makedirs(directory)
            print(f"Created new directory: {directory}")
            
        except Exception as e:
            print(f"Error processing directory {directory}: {str(e)}")
            raise

# Add to start of script, before other operations
initialize_directories()


# Load and validate JSON data
def load_json_data(input_file):
    """Load and validate JSON data from input file"""
    try:
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
            
        with open(input_file, 'r') as f:
            reviews_data = json.load(f)
            
        print(f"Successfully loaded {len(reviews_data)} reviews from {input_file}")
        return reviews_data
        
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON file: {str(e)}")
        raise
    except Exception as e:
        print(f"Error loading file: {str(e)}")
        raise

# Usage in main:
reviews_data = load_json_data(input_file)

def create_chunk_filename(chunk_number, base_name):
    """Create a filename for a chunk of reviews"""
    output_dir = "data-chunks"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = f"{base_name}_chunk_{chunk_number}.json"
    return os.path.join(output_dir, filename)

# Calculate total chunks to set for loop range
total_chunks = (len(reviews_data) + NUM_REVIEWS_PER_CHUNK - 1) // NUM_REVIEWS_PER_CHUNK

# Process chunks
# for i in range(total_chunks):
for i in range(min(NUM_CHUNKS, total_chunks)):
    start_idx = i * NUM_REVIEWS_PER_CHUNK
    end_idx = min((i + 1) * NUM_REVIEWS_PER_CHUNK, len(reviews_data))
    chunk_reviews = reviews_data[start_idx:end_idx]
    
    chunk_filename = create_chunk_filename(i + 1, base_name)
    
    with open(chunk_filename, 'w') as f:
        json.dump(chunk_reviews, f, indent=2)
    
    print(f"Processing chunk {i + 1} of {total_chunks}...")

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
            
            print(f"\nProcessing {chunk_file}...")
            print("\nReviews in this chunk:")
            print(f"{chunk_text}\n")
            print("-" * 80)  # Add visual separator between chunks

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
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }, f, indent=2)
            
        except Exception as e:
            print(f"Error processing {chunk_file}: {str(e)}")
            continue


# This function runs the process_chunks function and then compiles all analyzed files into one structured JSON file.
async def main():
    await process_chunks()

# This function compiles all analyzed files into one structured JSON file using the process_chunks function.
def compile_analyzed_files():
    """Compile all analyzed files into one structured JSON file"""
    analyzed_dir = "analyzed-chunks"
    output_dir = "summarized-reviews"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get all analyzed files
    analyzed_files = sorted([f for f in os.listdir(analyzed_dir) if f.startswith('analyzed_')])
    
    # Combine analyses
    combined_data = {
        "metadata": {
            "total_files": len(analyzed_files),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": "British Airways Trustpilot Reviews"
        },
        "analyses": []
    }
    
    # Process each file
    for file in analyzed_files:
        try:
            with open(os.path.join(analyzed_dir, file), 'r') as f:
                analysis = json.load(f)
                combined_data["analyses"].append({
                    "file": file,
                    "analysis": analysis["response"]
                })
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
    
    # Save combined file
    output_file = os.path.join(output_dir, f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(output_file, 'w') as f:
        json.dump(combined_data, f, indent=2)
    
    print(f"Combined analysis saved to {output_file}")

async def main():
    await process_chunks()
    compile_analyzed_files()

# Run the async function
if __name__ == "__main__":
    asyncio.run(main())