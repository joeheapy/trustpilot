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

