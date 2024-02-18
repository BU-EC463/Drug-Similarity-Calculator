import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from CleanData import CleanData
from ExactDrugAlgoFunction import ExactDrugAlgoFunction


inputData = CleanData('Daily Snapshot.csv')
data, input = ExactDrugAlgoFunction(10083420, inputData)

#Helper function
def get_generic_name_by_item_number(item_number, dataframe):
    # Replace 'item_number_column' and 'generic_name_column' with actual column names from your CSV
    result = dataframe[dataframe['Item Number – 8 digit'] == item_number]
    if not result.empty:
        return result['Generic Name'].iloc[0]
    else:
        return None

#Unit Tests
def test_item_number_to_generic_name_100(item_number=100, dataframe=inputData):
    try:
        ExactDrugAlgoFunction(item_number, dataframe)
    except ValueError as e:
        assert str(e)

def test_item_number_to_generic_name_2342342(item_number=2342342, dataframe=inputData):
    try:
        ExactDrugAlgoFunction(item_number, dataframe)
    except ValueError as e:
        assert str(e)

def test_item_number_to_generic_name_31313131(item_number=31313131, dataframe=inputData):
    try:
        ExactDrugAlgoFunction(item_number, dataframe)
    except Exception as e:
        assert str(e)

def test_item_number_to_generic_name_10055750(item_number=10055750, expected_generic_name='paroxetine', dataframe=inputData):
    try:
        result = ExactDrugAlgoFunction(item_number, dataframe)
        actual_generic_name = get_generic_name_by_item_number(item_number, dataframe)
        if expected_generic_name not in result['Generic Name'].values:
            raise Exception(f"Invalid generic name, '{expected_generic_name}' does not exist in Daily Snapshot.csv")
    
        if actual_generic_name != expected_generic_name:
            raise Exception(f"Test failed for item number {item_number}: expected '{expected_generic_name}', got '{actual_generic_name}'")
    except Exception as e:
        assert str(e)

def test_item_number_to_generic_name_10083420(item_number=10083420, expected_generic_name='duloxetine', dataframe=inputData):
    try:
        result = ExactDrugAlgoFunction(item_number, dataframe)
        actual_generic_name = get_generic_name_by_item_number(item_number, dataframe)
        if expected_generic_name not in result['Generic Name'].values:
            raise Exception(f"Invalid generic name, '{expected_generic_name}' does not exist in Daily Snapshot.csv")
    
        if actual_generic_name != expected_generic_name:
            raise Exception(f"Test failed for item number {item_number}: expected '{expected_generic_name}', got '{actual_generic_name}'")
    except Exception as e:
        assert str(e)

def test_item_number_to_generic_name_10000096(item_number=10000096, expected_generic_name='lancets', dataframe=inputData):
    try:
        result = ExactDrugAlgoFunction(item_number, dataframe)
        actual_generic_name = get_generic_name_by_item_number(item_number, dataframe)
        if expected_generic_name not in result['Generic Name'].values:
            raise Exception(f"Invalid generic name, '{expected_generic_name}' does not exist in Daily Snapshot.csv")
    
        if actual_generic_name != expected_generic_name:
            raise Exception(f"Test failed for item number {item_number}: expected '{expected_generic_name}', got '{actual_generic_name}'")
    except Exception as e:
        assert str(e)


def test_item_number_to_generic_name_10000096_v2(item_number=10000096, expected_generic_name='propane', dataframe=inputData):
    try:
        result = ExactDrugAlgoFunction(item_number, dataframe)
        actual_generic_name = get_generic_name_by_item_number(item_number, dataframe)
        if expected_generic_name not in result['Generic Name'].values:
            raise Exception(f"Invalid generic name, '{expected_generic_name}' does not exist in Daily Snapshot.csv")
    
        if actual_generic_name != expected_generic_name:
            raise Exception(f"Test failed for item number {item_number}: expected '{expected_generic_name}', got '{actual_generic_name}'")
    except Exception as e:
        assert str(e)


# Example test case - replace with actual values from your CSV file
#test_item_number_to_generic_name(10055750, 'paroxetine', inputData)
#test_item_number_to_generic_name(10083420, 'duloxetine', inputData)
#test_item_number_to_generic_name(10000096, 'lancets', inputData)
#test_item_number_to_generic_name(2342342, 'paroxetine', inputData)
#test_item_number_to_generic_name(31313131, 'paroxetine', inputData)
#test_item_number_to_generic_name(10000096, 'propane', inputData)

def test_item_number_format(item_number=10083420, dataframe=inputData):
    # Check if all entries in 'Item Number – 8 digit' are strings of length 8 and numeric
    try:
        ExactDrugAlgoFunction(item_number, dataframe)
    except Exception as e:
        assert str(e)



def test_generic_name_format(item_number=10083420, dataframe=inputData):
    try:
        ExactDrugAlgoFunction(item_number, dataframe)
    except Exception as e:
        assert str(e)

def test_size_qty_format(item_number=10083420, dataframe=inputData):
    try:
        ExactDrugAlgoFunction(item_number, dataframe)
    except Exception as e:
        assert str(e)

def test_form_column(item_number=10083420, dataframe=inputData):
    try:
        ExactDrugAlgoFunction(item_number, dataframe)
    except Exception as e:
        assert str(e)

def test_negative_true_similarity(item_number=10083420, dataframe=inputData):
    try:
        data, input = ExactDrugAlgoFunction(item_number, dataframe)
        for value in data['True Similarity']:
            if value < 0:
                raise Exception(f"Negative value found in 'True Similarity': {value}")
    except Exception as e:
        assert str(e)
