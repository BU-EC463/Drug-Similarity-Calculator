import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from CleanData import CleanData
from ExactDrugAlgoFunction import ExactDrugAlgoFunction


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
#test_item_number_to_generic_name(10055750, 'paroxetine', inputData)
#test_item_number_to_generic_name(10083420, 'duloxetine', inputData)
#test_item_number_to_generic_name(10000096, 'lancets', inputData)
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

def test_generic_name_format(dataframe):
    for item in dataframe['Generic Name']:
        assert isinstance(item, str), "Non-string value found in 'Generic Name'"
    print("All generic names are properly formatted.")


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

def test_negative_true_similarity(dataframe):
    for value in dataframe['True Similarity']:
        assert value >= 0, f"Negative value found in 'True Similarity': {value}"
    print("No negative value found in 'True Similarity'")
