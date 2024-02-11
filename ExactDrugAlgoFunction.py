

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def ExactDrugAlgoFunction(drug_code, data, expected_generic_name=None):

    #Helper function for unit test
    def get_generic_name_by_item_number(item_number, dataframe):
        # Replace 'item_number_column' and 'generic_name_column' with actual column names from your CSV
        result = dataframe[dataframe['Item Number – 8 digit'] == item_number]
        if not result.empty:
            return result['Generic Name'].iloc[0]
        else:
            return None

    actual_generic_name = get_generic_name_by_item_number(drug_code, data)
    if expected_generic_name != None:
        if expected_generic_name not in data['Generic Name'].values:
            raise Exception(f"Invalid generic name, '{expected_generic_name}' does not exist in Daily Snapshot.csv")
    
        if actual_generic_name != expected_generic_name:
            raise Exception(f"Test failed for item number {drug_code}: expected '{expected_generic_name}', got '{actual_generic_name}'")

    if len(str(drug_code)) != 8:
        raise ValueError("Drug code must be exactly 8 digits long.")
    
    if drug_code not in data['Item Number – 8 digit'].values:
        raise Exception(f"Invalid item number, '{drug_code}' does not exist in Daily Snapshot.csv")
    
    for item in data['Size Qty']:
        if isinstance(item, int):
            raise Exception(f"String value found in 'Size Qty'")
    
    for item in data['Item Number – 8 digit']:
        item_str = str(item)
        if not isinstance(item, int):
            raise Exception(f"String value found in 'Item Number – 8 digit'")
        if len(item_str) != 8:
            raise Exception(f"Found an item with incorrect length: {item}")
        
    for item in data['Generic Name']:
        if not isinstance(item, str):
            raise Exception(f"Non-string value found in 'Generic Name'")

    for index, row in data.iterrows():
        form_value = row['Form']
        word_count = len(form_value.split())
        if word_count > 7:
            raise Exception(f"Entry in 'Form' column exceeds 7 words at index {index}: '{form_value}'")

    # Item number of the drug to run the similarity test on
    reference_item_number = drug_code

    # Find and print the row for the given reference_item_number
    matching_row = data[data['Item Number – 8 digit'] == reference_item_number]
    

    # Find the Generic Name for the given reference_item_number
    reference_generic_name = data.loc[data['Item Number – 8 digit'] == reference_item_number, 'Generic Name'].iloc[0]


    # Create a copy of the dataframe filtered by Generic Name
    data_generic = data[data['Generic Name'] == reference_generic_name].copy()

    # Selecting the reference item
    reference_item = data_generic[data_generic['Item Number – 8 digit'] == reference_item_number]
    if reference_item.empty:
        return "Reference item not found in the dataset."

    # Extracting the form of the reference item
    reference_form = reference_item.iloc[0]['Form']
    forms = data_generic['Form'].tolist()
    forms.insert(0, reference_form)

    # Vectorizing the forms using TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(forms)

    # Calculating cosine similarity
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    data_generic['Similarity'] = cosine_similarities

    # Filtering the dataset to show only items with a similarity score above 0.9
    similarity_items = data_generic[data_generic['Similarity'] > 0].copy()

    # Extracting the size of the reference item
    reference_size = reference_item.iloc[0]['Size']
    sizes = similarity_items['Size'].tolist()
    sizes.insert(0, reference_size)

    # Vectorizing the sizes using TF-IDF
    tfidf_matrix_sizes = vectorizer.fit_transform(sizes)

    # Calculating cosine similarity for sizes
    cosine_similarities_sizes = cosine_similarity(tfidf_matrix_sizes[0:1], tfidf_matrix_sizes[1:]).flatten()
    similarity_items['Size Similarity'] = cosine_similarities_sizes

    # Remove input item
    similarity_items = similarity_items[similarity_items['Item Number – 8 digit'] != reference_item_number]

    # Define true similarity
    w1 = 1
    w2 = 1
    similarity_items['True Similarity'] = (w1 * similarity_items['Similarity'] + w2 * similarity_items['Size Similarity']) / (w1 + w2)

    # Cleaning and sort the data
    similarity_items = similarity_items.drop(columns=['Similarity', 'Size Similarity'])
    similarity_items = similarity_items.sort_values(by=['True Similarity'], ascending=False)
    
    
    return similarity_items, matching_row