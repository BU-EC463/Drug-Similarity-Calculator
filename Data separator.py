def convert_to_csv_with_columns(input_file_path, output_file_path):
    import pandas as pd

    # Define the column names based on the image provided
    column_names = [
        'Item Number – 6 digit', 'NDC Number', 'UPC Number', 'Constant',
        'Customer-Specific Item Number', 'Description', 'Pack Size Divisor',
        'Size Qty', 'RX/OTC Indicator', 'AWP Price', 'Acquisition Price',
        'Retail Price', 'Contract Flag', 'Generic Description',
        'Retail Pack Quantity', 'WAC Price', 'Item Number – 8 digit'
    ]
    
    # Read the data using pandas, treating all data as strings
    df = pd.read_csv(input_file_path, sep='|', header=None, names=column_names, dtype=str)
    
    # Output the dataframe to a CSV file without index and with the header
    df.to_csv(output_file_path, index=False, header=column_names)

convert_to_csv_with_columns('BIDMC WEST GPO1003155131114.TXT', 'Daily Snapshot.csv')
