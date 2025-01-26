import pandas as pd
import os
from openai import OpenAI
from datetime import datetime
import json
import asyncio
from dotenv import load_dotenv
from functions import get_input_file, process_chunks, initialize_directories, generate_journey_steps, map_reviews_to_journey, plot_average_ratings

# Set chunk size
NUM_REVIEWS_PER_CHUNK = 10
# Set number of chunks to process
NUM_CHUNKS = 9

# Load environment variables
load_dotenv()

# Loads the raw source data from the input file and returns it as a list of dictionaries.
input_file, base_name = get_input_file()

# Get base name from input file
base_name = os.path.splitext(os.path.basename(input_file))[0]

# This function deletes and recreates the working directories to ensure a clean start
initialize_directories()

# Load and validate JSON data
def load_json_data(input_file):
    """Load and validate JSON data from input file"""
    try:
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
            
        with open(input_file, 'r') as f:
            reviews_data = json.load(f)
            
        print(f"Successfully loaded {len(reviews_data):,} reviews")
        print(f"Processing {NUM_CHUNKS:,} batches of {NUM_REVIEWS_PER_CHUNK:,} reviews each")
        print(f"Total reviews to process: {NUM_CHUNKS * NUM_REVIEWS_PER_CHUNK:,}")
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
    
    print(f"Making batch {i + 1} of {NUM_CHUNKS}...")

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
        # "metadata": {
        #     "total_files": len(analyzed_files),
        #     "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        # },
        "analyses": []
    }
    
    # Process each file
    for file in analyzed_files:
        try:
            with open(os.path.join(analyzed_dir, file), 'r') as f:
                analysis = json.load(f)
                combined_data["analyses"].append({
                    "analysis": analysis["response"]
                })
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
    
    # Save combined file
    output_file = os.path.join(output_dir, f"summarized_reviews_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(output_file, 'w') as f:
        json.dump(combined_data, f, indent=2)


async def main():
    try:
        # Process reviews in chunks
        await process_chunks()
        
        # Compile all analyzed files
        compile_analyzed_files()
        
        # Generate customer journey steps
        await generate_journey_steps()
        print("\nCustomer journey analysis complete")
        
        # Map reviews to journey
        await map_reviews_to_journey()
        print("\nReview journey mapping complete")
        
        # Generate and save plot
        plot_average_ratings()
        print("\nPlotting complete")
        
    except Exception as e:
        print(f"\nError in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())