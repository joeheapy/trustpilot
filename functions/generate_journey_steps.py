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
    """Generate customer journey steps from ssummarized-reviews"""
    try:
        # Find latest sentiment analysis file
        summary_dir = "summarized-reviews"
        if not os.path.exists(summary_dir):
            raise FileNotFoundError(f"Directory not found: {summary_dir}")
        
        files = sorted([f for f in os.listdir(summary_dir) 
                       if f.startswith('summarized_reviews_')])
        
        if not files:
            raise FileNotFoundError("No sentiment analysis files found")
        
        latest_file = os.path.join(summary_dir, files[-1])
        print(f"\nSending summarised reviews to OpenAI to determine journey steps: {latest_file}")
        
        # Read sentiment analysis data
        with open(latest_file, 'r') as f:
            analysis_data = json.load(f)
        
        # Set up journey analysis prompt
        journey_prompt = """Review the provided data to determine the type of service the company offers. Identify 10 steps in a typical customer journey, starting when a potential customer becomes aware of the product or service through decision-making, purchase, using the product or service, and following up.

        Output:
        • Provide a descriptive list of named customer journey stages. 
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
                    "step_name": "name",
                    "description": "brief description"
                }
            ]
        }"""
        
           # Make OpenAI API call
        response = await client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are analyzing customer journey data. Return only valid JSON."},
                {"role": "user", "content": json.dumps(analysis_data)},
                {"role": "user", "content": journey_prompt}
            ]
        )
        
        # This is the response from the OpenAI API
        content = response.choices[0].message.content.strip()

        try:
            # Clean and parse response
            cleaned_content = content.replace('\n', '').replace('```json', '').replace('```', '')
            journey_data = json.loads(cleaned_content)
            
            if "journey_steps" not in journey_data:
                raise ValueError("Response missing journey_steps key")
                
        except json.JSONDecodeError as e:
            print(f"JSON Parse Error: {str(e)}\nContent: {content}")
            raise
        
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
        
        print(f"\nJourney steps saved to: {output_file}")
        return journey_data
        
    except Exception as e:
        print(f"Error generating journey steps: {str(e)}")
        raise