# Sentiment Analysis Tool

Analyzes customer reviews using OpenAI API to map customer journey and sentiment.

## Installation

```bash
git clone https://github.com/username/sentiment-analysis.git
cd sentiment-analysis
python -m venv .venv
source .venv/bin/activate  # On Mac/Linux
pip install -r requirements.txt
```

## Structure

sentiment-analysis/
├── functions/
│ ├── **init**.py
│ ├── get_input_file.py
│ ├── process_chunks.py
│ ├── clean_markdown.py
│ ├── initialize_directories.py
│ ├── generate_journey_steps.py
│ ├── map_reviews_to_journey.py
│ └── plot_average_ratings.py
├── main.py
├── requirements.txt
└── setup.py

## Main Execution (main.py)

```python
# filepath: /Users/joeheapy/Documents/sentiment-analysis/main.py

import asyncio
import os
from datetime import datetime
import json
from functions import (
    get_input_file,
    process_chunks,
    initialize_directories,
    generate_journey_steps,
    map_reviews_to_journey,
    plot_average_ratings
)

NUM_REVIEWS_PER_CHUNK = 10
NUM_CHUNKS = 9

async def main():
    try:
        # Process reviews in chunks
        await process_chunks()

        # Compile analyzed files
        compile_analyzed_files()

        # Generate customer journey steps
        await generate_journey_steps()
        print("\nCustomer journey analysis complete")

        # Map reviews to journey
        await map_reviews_to_journey()
        print("\nReview journey mapping complete")

        # Generate and save plot
        await plot_average_ratings()
        print("\nPlotting complete")

    except Exception as e:
        print(f"\nError in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
```

## Function: clean_markdown()

```python
# filepath: /Users/joeheapy/Documents/sentiment-analysis/functions/clean_markdown.py

import re

def clean_markdown(text):
    """Remove markdown formatting characters from text"""

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
```

# filepath: initialize_directories()

```python

import shutil
from datetime import datetime
import os

# This deletes and recreates the working directories to ensure a clean start
def initialize_directories():
    """Initialize working directories by removing and recreating them"""
    directories = [
        'analyzed-chunks',
        'data-chunks',
        'summarized-reviews',
        'journey-steps',
        'reviews-by-journey-step',
        'visualizations'
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
```

## Function: get_input_file()

```python

import os

def get_input_file() -> tuple[str, str]:
    """Get the input file path and base name from raw_trustpilot_data directory"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "raw_trustpilot_data")

    try:
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            print(f"Created directory: {data_dir}")
            raise FileNotFoundError(f"No JSON files found in {data_dir}")

        json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]

        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {data_dir}")

        input_file = os.path.join(data_dir, json_files[0])
        base_name = os.path.splitext(os.path.basename(input_file))[0]

        print(f"Found input file: {input_file}")
        return input_file, base_name

    except Exception as e:
        print(f"Error finding input file: {str(e)}")
        raise
```

## Function: process_chunks()

```python

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
```

## Function: generate_journey_steps()

```python

import os
import json
from datetime import datetime
from dotenv import load_dotenv
from openai import AsyncOpenAI

# Load environment variables
load_dotenv()

# Initialize AsyncOpenAI client
client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))

async def generate_journey_steps():
    """Generate customer journey steps from summarized reviews"""
    try:
        # Find latest sentiment analysis file
        summary_dir = "summarized-reviews"
        if not os.path.exists(summary_dir):
            raise FileNotFoundError(f"Directory not found: {summary_dir}")

        files = sorted([f for f in os.listdir(summary_dir)
                       if f.startswith('summarized_reviews_')])

        if not files:
            raise FileNotFoundError("No sentiment analysis files found")

        summarized_reviews_file = os.path.join(summary_dir, files[-1])
        print(f"Found latest analysis file: {summarized_reviews_file}")

        # Read and clean sentiment analysis data
        with open(summarized_reviews_file, 'r') as f:
            content = f.read()
            # Remove both actual newlines and escaped newlines
            cleaned_content = content.replace('\n', '').replace('\\n', '')
            analysis_data = json.loads(cleaned_content)

        if not analysis_data:
            raise ValueError("Empty or invalid analysis data")

        # Set up journey analysis prompt
        journey_prompt = """Review the provided data to determine the type of service the company offers. Identify 10 steps in a typical customer journey, starting when a potential customer becomes aware of the product or service through decision-making, purchase, using the product or service, and following up.

        Output:
        • Provide a descriptive list of named customer journey stages that are specific to this service or product.
        • Ensure several of the steps describe the customer's use of the product or service.
        • DO NOT includ "feedback" as a step.
        • Title each step to reflect its relevance to the service offered.
        • Capture every significant stage in the journey comprehensively.
        • Ensure you have identified 10 distinct steps.

        Please return the response in this JSON format:
        {
            "journey_steps": [
                {
                    "step_number": 1,
                    "step_name": "step name",
                    "description": "description of this step"
                }
            ]
        }"""

        # Make OpenAI API call
        response = await client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are analyzing customer journey data."},
                {"role": "user", "content": json.dumps(analysis_data)},
                {"role": "user", "content": journey_prompt}
            ]
        )

        # Parse and validate response
        # content = response.choices[0].message.content.strip()
        # print(f"Raw response content: {content[:200]}...")

        try:
            journey_data = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"JSON Parse Error: {str(e)}\nContent: {content}")
            raise

        if "journey_steps" not in journey_data:
            raise ValueError("Response missing journey_steps key")

        # Save journey steps
        journey_dir = "journey-steps"
        if not os.path.exists(journey_dir):
            os.makedirs(journey_dir)

        output_file = os.path.join(
            journey_dir,
            f"customer_journey_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        with open(output_file, 'w') as f:
            json.dump({
                "journey_steps": journey_data["journey_steps"]
            }, f, indent=2)

        print(f"Journey steps saved to: {output_file}")
        return journey_data

    except Exception as e:
        print(f"Error generating journey steps: {str(e)}")
        raise
```

## Function: map_reviews_to_journey()

````python

import os
import json
from datetime import datetime
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Load environment variables and initialize client
load_dotenv()
client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))


def convert_date_format(date_str):
    """Convert various date formats to YYYY-MM-DD"""
    try:
        # Try parsing common date formats
        for fmt in [
            "%B %d, %Y",      # January 17, 2025
            "%d %B %Y",       # 17 January 2025
            "%Y-%m-%d",       # 2025-01-17
            "%d/%m/%Y"        # 17/01/2025
        ]:
            try:
                return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
            except ValueError:
                continue
        raise ValueError(f"Unsupported date format: {date_str}")
    except Exception as e:
        raise ValueError(f"Date conversion error: {date_str} - {str(e)}")


async def map_reviews_to_journey():
    """Map reviews to customer journey steps"""
    try:
        # Find latest files
        summary_dir = "summarized-reviews"
        journey_dir = "journey-steps"
        target_dir = "reviews-by-journey-step"

        # Get latest summarized reviews
        summary_files = sorted([f for f in os.listdir(summary_dir)
                              if f.startswith('summarized_reviews_')])
        if not summary_files:
            raise FileNotFoundError("No summarized reviews found")
        latest_summary = os.path.join(summary_dir, summary_files[-1])

        # Get latest journey steps
        journey_files = sorted([f for f in os.listdir(journey_dir)
                              if f.startswith('customer_journey_')])
        if not journey_files:
            raise FileNotFoundError("No journey steps found")
        latest_journey = os.path.join(journey_dir, journey_files[-1])

        # Load and validate source files
        with open(latest_summary, 'r') as f:
            reviews_data = json.load(f)
        with open(latest_journey, 'r') as f:
            journey_data = json.load(f)

        # Validate journey data structure
        if not journey_data.get('journey_steps'):
            raise ValueError("Journey data missing journey_steps")

        # Create set of valid step names for validation
        valid_steps = {step['step_name'] for step in journey_data['journey_steps']}

        mapping_prompt = """Map each review to the most relevant customer journey step.

        Rules:
        1. Use ONLY the journey steps provided - do not create new steps
        2. Match each review to exactly one journey step
        3. Convert dates to YYYY-MM-DD format (e.g., 2028-01-17)
        4. Use exact step names from the journey steps list

        Return ONLY this JSON structure:
        {
            "reviews_by_journey_step": [
                {
                    "step_name": "exact step name from journey steps",
                    "rating": 5,
                    "reviewDateOfExperience": "YYYY-MM-DD"
                }
            ]
        }

        Requirements:
        - reviewDateOfExperience must match original review date exactly and in the format YYYY-MM-DD (e.g., 2028-01-17)
        - rating must be integer 1-5
        - step_name must exactly match one from provided journey steps
        - all fields are required"""

        # Make OpenAI API call
        response = await client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": "You are mapping customer reviews to journey steps."},
                {"role": "user", "content": f"Journey steps: {json.dumps(journey_data)}"},
                {"role": "user", "content": f"Reviews: {json.dumps(reviews_data)}"},
                {"role": "user", "content": mapping_prompt}
            ]
        )

        # Clean and validate response
        content = response.choices[0].message.content.strip()
        # print(f"Raw response: {content[:200]}...")  # Debug log

        # Remove markdown code block markers
        content = content.replace('```json', '').replace('```', '').strip()
        # print(f"Cleaned content: {content[:200]}...")  # Debug log

        try:
            mapped_data = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"JSON Parse Error: {str(e)}\nContent: {content}")
            raise

        # Validate structure
        if "reviews_by_journey_step" not in mapped_data:
            raise ValueError("Response missing reviews_by_journey_step key")

        # Validate reviews
        for review in mapped_data["reviews_by_journey_step"]:
            required_fields = ["step_name", "rating", "reviewDateOfExperience"]
            missing_fields = [f for f in required_fields if f not in review]
            if missing_fields:
                raise ValueError(f"Review missing required fields: {missing_fields}")

            # Convert and validate date
            try:
                review["reviewDateOfExperience"] = convert_date_format(review["reviewDateOfExperience"])
            except ValueError as e:
                raise ValueError(f"Date format error: {str(e)}")

            # Validate field types and values
            if review["step_name"] not in valid_steps:
                raise ValueError(f"Invalid step_name: {review['step_name']}")

            if not isinstance(review["rating"], int) or not 1 <= review["rating"] <= 5:
                raise ValueError(f"Invalid rating value: {review['rating']}")

            try:
                datetime.strptime(review["reviewDateOfExperience"], "%Y-%m-%d")
            except ValueError:
                raise ValueError(f"Invalid date format: {review['reviewDateOfExperience']}")

        # Save mapped reviews
        output_file = os.path.join(
            target_dir,
            f"journey_mapped_reviews_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        with open(output_file, 'w') as f:
            json.dump({
                "reviews_by_journey_step": mapped_data["reviews_by_journey_step"]
            }, f, indent=2)

        print(f"\nMapped reviews saved to: {output_file}")
        return mapped_data

    except Exception as e:
        print(f"/nError mapping reviews to journey: {str(e)}")
        raise

````

## Function: plot_average_ratings()

```python

import os
import json
import glob
from datetime import datetime
import pandas as pd
import plotly.express as px

def get_latest_file(pattern):
    """Get the most recent file matching the pattern"""
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No files found matching pattern: {pattern}")
    return max(files, key=os.path.getctime)

def validate_data(data, required_keys):
    """Validate JSON data structure"""
    if not isinstance(data, dict):
        raise ValueError("Data must be a dictionary")
    missing_keys = [key for key in required_keys if key not in data]
    if missing_keys:
        raise ValueError(f"Missing required keys: {missing_keys}")

def plot_average_ratings():
    """Generate interactive plot of average ratings by journey step"""
    try:
        # Find latest files
        reviews_file = get_latest_file("reviews-by-journey-step/journey_mapped_reviews_*.json")
        journey_file = get_latest_file("journey-steps/customer_journey_*.json")

        # Load and validate reviews data
        with open(reviews_file, 'r') as f:
            reviews_data = json.load(f)
        validate_data(reviews_data, ["reviews_by_journey_step"])

        # Load and validate journey steps
        with open(journey_file, 'r') as f:
            steps_data = json.load(f)
        validate_data(steps_data, ["journey_steps"])

        # Get required steps
        required_steps = [step['step_name'] for step in steps_data['journey_steps']]
        if not required_steps:
            raise ValueError("No journey steps found")

        # Create DataFrame
        reviews_df = pd.json_normalize(reviews_data['reviews_by_journey_step'])
        if reviews_df.empty:
            raise ValueError("No review data found")

        # Validate required columns
        required_columns = ['step_name', 'rating']
        missing_columns = [col for col in required_columns if col not in reviews_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Map ratings
        mapping = {5: 2, 4: 1, 3: 0, 2: -1, 1: -2}
        reviews_df['rating'] = reviews_df['rating'].map(mapping)

        # Calculate averages
        average_ratings = (reviews_df.groupby('step_name')['rating']
                         .mean()
                         .reset_index())

        # Reorder based on journey steps
        average_ratings = (average_ratings.set_index('step_name')
                         .reindex(required_steps, fill_value=0)
                         .reset_index())

        # Create plot
        fig = px.bar(
            average_ratings,
            x='step_name',
            y='rating',
            title='Average Rating By Journey Step',
            labels={'step_name': 'Journey Step', 'rating': 'Average Rating'},
            category_orders={'step_name': required_steps}
        )

        # Customize plot
        fig.update_traces(marker_color='skyblue')
        fig.update_yaxes(tickvals=[-2, -1, 0, 1, 2])
        fig.update_layout(
            xaxis_tickangle=-45,
            yaxis_title="Average Rating (-2 to +2)",
            xaxis_title="Journey Step"
        )

        # Save plot
        output_dir = "visualizations"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_file = os.path.join(
            output_dir,
            f"ratings_by_step_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        )
        fig.write_html(output_file)
        print(f"\nPlot saved to: {output_file}")

        # Display plot
        fig.show()

    except Exception as e:
        print(f"/nError generating plot: {str(e)}")
        raise

```
