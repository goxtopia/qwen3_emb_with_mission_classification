from datasets import load_dataset
import random
import pandas as pd

# Load the dataset using Hugging Face datasets
dataset = load_dataset('heyanzhuo/glaive-function-calling-v2-parsed')

# Initialize lists to store the processed data
processed_data = []

false = False
true = True

# Process each item in the dataset
for item in dataset['train']:
    query = item['history']
    tools = item['all_funcs']
    answers = item['func']

    if len(query) > 4 * 4000:
        # about 8000 words - 8000 or less tokens
        continue

    try:
        tools = eval(tools)  # Convert string representation of list to actual list
    except Exception as e:
        print(f"Error evaluating tools for query '{query}': {e}")
        continue
    
    func_name = answers

    # Create positive examples (label = 1)
    for tool in tools:
        function_name = tool['name']
        function_description = tool['description']

        if function_name != func_name:
            continue
        
        # Positive example
        processed_data.append({
            'text1': str(query),
            'text2': str(tool),
            'label': 1
        })
        
        # Create negative example (label = 0)
        # Randomly select a different function description from the same tools list
        other_tools = [t for t in tools if t['name'] != function_name]
        for tool in other_tools:
            random_tool = tool
            random_description = random_tool['description']
            
            processed_data.append({
                'text1': str(query),
                'text2': str(tool),
                'label': 0
            })

# Convert to DataFrame for CSV output
result_df = pd.DataFrame(processed_data)

# Save to CSV
result_df.to_csv('processed_dataset_mul_turn.csv', index=False)

# Print sample output
print(result_df.head())
