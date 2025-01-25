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
                    "date": "YYYY-MM-DD"
                }
            ]
        }

        Requirements:
        - date must match original review date exactly and in the format YYYY-MM-DD (e.g., 2028-01-17)
        - rating must be integer 1-5
        - step_name must exactly match one from provided journey steps
        - all fields are required"""
        
        # Make OpenAI API call
        response = await client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are mapping customer reviews to journey steps."},
                {"role": "user", "content": f"Journey steps: {json.dumps(journey_data)}"},
                {"role": "user", "content": f"Reviews: {json.dumps(reviews_data)}"},
                {"role": "user", "content": mapping_prompt}
            ]
        )
        
        # Validate and parse response
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty response from OpenAI")
            
        try:
            mapped_data = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"Invalid JSON response: {content}")
            raise
        
        # Validate response structure
        if "reviews_by_journey_step" not in mapped_data:
            raise ValueError("Response missing reviews_by_journey_step")
            
        # Validate each review
        for review in mapped_data["reviews_by_journey_step"]:
            # Check required fields
            required_fields = ["step_name", "rating", "date"]
            missing_fields = [f for f in required_fields if f not in review]
            if missing_fields:
                raise ValueError(f"Review missing required fields: {missing_fields}")
            
            # Convert and validate date
            try:
                review["date"] = convert_date_format(review["date"])
            except ValueError as e:
                raise ValueError(f"Date format error: {str(e)}")
            
            # Validate field types and values
            if review["step_name"] not in valid_steps:
                raise ValueError(f"Invalid step_name: {review['step_name']}")

            if not isinstance(review["rating"], int) or not 1 <= review["rating"] <= 5:
                raise ValueError(f"Invalid rating value: {review['rating']}")
                
            try:
                datetime.strptime(review["date"], "%Y-%m-%d")
            except ValueError:
                raise ValueError(f"Invalid date format: {review['date']}")
        
        # Save mapped reviews
        output_file = os.path.join(
            target_dir, 
            f"journey_mapped_reviews_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        with open(output_file, 'w') as f:
            json.dump({
                "metadata": {
                    "summary_source": latest_summary,
                    "journey_source": latest_journey,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                },
                "mapped_reviews": mapped_data["reviews_by_journey_step"]
            }, f, indent=2)
        
        print(f"Mapped reviews saved to: {output_file}")
        return mapped_data
        
    except Exception as e:
        print(f"Error mapping reviews to journey: {str(e)}")
        raise
