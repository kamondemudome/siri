import pandas as pd

# Load the new datasets
new_data1 = pd.read_csv('new_data1.csv')
new_data2 = pd.read_csv('new_data2.csv')

# Print column names
print("Columns in new_data1:", new_data1.columns.tolist())
print("Columns in new_data2:", new_data2.columns.tolist())

# Print data types
print("\nData types in new_data1:\n", new_data1.dtypes)
print("\nData types in new_data2:\n", new_data2.dtypes)

# Print row counts
print(f"\nRows in new_data1: {len(new_data1)}")
print(f"Rows in new_data2: {len(new_data2)}")

# Check class distribution
print("\nClass distribution in new_data1 (Diabetes_012):\n", new_data1['Diabetes_012'].value_counts(normalize=True))
print("\nClass distribution in new_data2 (Diabetes_binary):\n", new_data2['Diabetes_binary'].value_counts(normalize=True))

# Load the original dataset
original_df = pd.read_csv('new_dataset.csv')

# Ensure the original dataset has the correct column names
original_df.columns = ['Diabetes_binary', 'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 
                       'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies', 
                       'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth', 'MentHlth', 
                       'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income']

# Convert Diabetes_012 in new_data1 to Diabetes_binary
# 0, 1 -> 0 (no diabetes or prediabetes); 2 -> 1 (diabetes)
new_data1['Diabetes_binary'] = new_data1['Diabetes_012'].apply(lambda x: 1 if x == 2 else 0)
new_data1 = new_data1.drop(columns=['Diabetes_012'])

# Ensure new_data2 has the correct column names (already has Diabetes_binary)
new_data2.columns = ['Diabetes_binary', 'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 
                     'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies', 
                     'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth', 'MentHlth', 
                     'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income']

# Combine the datasets
combined_df = pd.concat([original_df, new_data1, new_data2], ignore_index=True)

# Save the combined dataset
combined_df.to_csv('combined_dataset.csv', index=False)

# Print summary
print(f"\nOriginal dataset rows: {len(original_df)}")
print(f"New_data1 rows: {len(new_data1)}")
print(f"New_data2 rows: {len(new_data2)}")
print(f"Combined dataset rows: {len(combined_df)}")
print("\nClass distribution in combined dataset:\n", combined_df['Diabetes_binary'].value_counts(normalize=True))