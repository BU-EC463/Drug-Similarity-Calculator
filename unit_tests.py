import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer

def CleanData(file_path):
    # Reading the file

    # Load the data from the CSV file
    data = pd.read_csv(file_path)

    # Removing the specified columns and reordering the 'Item Number – 8 digit' column
    columns_to_remove = ['Item Number – 6 digit', 'UPC Number', 'Constant', 
                        'Customer-Specific Item Number', 'Pack Size Divisor', 
                        'RX/OTC Indicator']

    # Removing the columns
    data_cleaned = data.drop(columns=columns_to_remove)

    # Reordering 'Item Number – 8 digit' to the left
    column_to_move = data_cleaned.pop('Item Number – 8 digit')
    data_cleaned.insert(0, 'Item Number – 8 digit', column_to_move)

    # Moving all price columns and the contract flag to the right
    columns_to_move = ['AWP Price', 'Acquisition Price', 'Retail Price', 'WAC Price', 'Contract Flag']
    for col in columns_to_move:
        data_cleaned[col] = data_cleaned.pop(col)

    import re

    # Function to split the generic description into generic name and form
    def split_description(desc):
        match = re.search(r'[A-Z]', desc)
        if match:
            index = match.start()
            return desc[:index].strip(), desc[index:].strip()
        else:
            return desc, ''

    # Applying the function to split 'Generic Description'
    data_cleaned['Generic Name'], data_cleaned['Form'] = zip(*data_cleaned['Generic Description'].apply(split_description))
    data_cleaned.drop(columns=['Generic Description'], inplace=True)

    # Removing rows where 'Generic Name' is empty or whitespace
    data_cleaned = data_cleaned[data_cleaned['Generic Name'].str.strip() != '']

    # Function to split the description into name and size
    def split_description_on_number(desc):
        match = re.search(r'\d', desc)
        if match:
            index = match.start()
            return desc[:index].strip(), desc[index:].strip()
        else:
            return desc, ''

    # Applying the function to split 'Description'
    data_cleaned['Name'], data_cleaned['Size'] = zip(*data_cleaned['Description'].apply(split_description_on_number))
    data_cleaned.drop(columns=['Description'], inplace=True)

    return data_cleaned

def ExactDrugAlgoFunction(drug_code, data):
    # Item number of the drug to run the similarity test on
    reference_item_number = drug_code

    # Find and print the row for the given reference_item_number
    matching_row = data[data['Item Number – 8 digit'] == reference_item_number]
    

    # Find the Generic Name for the given reference_item_number
    reference_generic_name = data.loc[data['Item Number – 8 digit'] == reference_item_number, 'Generic Name'].iloc[0]


    # Create a copy of the dataframe filtered by Generic Name
    data_generic = data[data['Generic Name'] == reference_generic_name].copy()

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

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


inputData = CleanData('Daily Snapshot.csv')
data, input = ExactDrugAlgoFunction(10083420, inputData)

#Unit Tests
def get_generic_name_by_item_number(item_number, dataframe):
    # Replace 'item_number_column' and 'generic_name_column' with actual column names from your CSV
    result = dataframe[dataframe['Item Number – 8 digit'] == item_number]
    if not result.empty:
        return result['Generic Name'].iloc[0]
    else:
        return None

def test_item_number_to_generic_name(item_number, expected_generic_name, dataframe):
    assert len(str(item_number)) == 8, f"Item number isn't 8"
    assert item_number in dataframe['Item Number – 8 digit'].values, f"Invalid item number, '{item_number}' does not exist in Daily Snapshot.csv"
    assert expected_generic_name in dataframe['Generic Name'].values, f"Invalid generic name, '{expected_generic_name}' does not exist in Daily Snapshot.csv"
    actual_generic_name = get_generic_name_by_item_number(item_number, dataframe)
    assert actual_generic_name == expected_generic_name, f"Test failed for item number {item_number}: expected '{expected_generic_name}', got '{actual_generic_name}'"
    print(f"Test passed for item number {item_number}")


# Example test case - replace with actual values from your CSV file
test_item_number_to_generic_name(10055750, 'paroxetine', inputData)
test_item_number_to_generic_name(10083420, 'duloxetine', inputData)
test_item_number_to_generic_name(10000096, 'lancets', inputData)
#test_item_number_to_generic_name(2342342, 'paroxetine', inputData)
#test_item_number_to_generic_name(31313131, 'paroxetine', inputData)
#test_item_number_to_generic_name(10000096, 'propane', inputData)

def test_item_number_format(dataframe):
    # Check if all entries in 'Item Number – 8 digit' are strings of length 8 and numeric
    for item in dataframe['Item Number – 8 digit']:
        item_str = str(item)
        assert isinstance(item, int), "String value found in 'Item Number – 8 digit'"
        assert len(item_str) == 8, f"Found an item with incorrect length: {item}"
    print("All 8-digit identifiers are properly formatted.")

# Run the test
test_item_number_format(inputData)

def test_generic_name_format(dataframe):
    for item in dataframe['Generic Name']:
        assert isinstance(item, str), "Non-string value found in 'Generic Name'"
    print("All generic names are properly formatted.")


test_generic_name_format(inputData)

'''
def test_size_qty_format(dataframe):
    for item in dataframe['Size Qty']:
        assert isinstance(item, int), "String value found in 'Size Qty'"
    print("All size qty are properly formatted.")

test_size_qty_format(inputData)
'''

def test_form_column(dataframe):
    for index, row in dataframe.iterrows():
        form_value = row['Form']
        word_count = len(form_value.split())
        assert word_count <= 7, f"Entry in 'Form' column exceeds 7 words at index {index}: '{form_value}'"
    print("Every form is less than 7 words")

test_form_column(inputData)

def test_negative_true_similarity(dataframe):
    for value in dataframe['True Similarity']:
        assert value >= 0, f"Negative value found in 'True Similarity': {value}"
    print("No negative value found in 'True Similarity'")
test_negative_true_similarity(data)