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