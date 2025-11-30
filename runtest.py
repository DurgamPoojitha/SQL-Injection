import pandas as pd
import time
import sys

# --- IMPORT YOUR DETECTION TOOL ---
try:
    from demofinal import sql_injection_detection
except ImportError:
    print("\nCRITICAL ERROR: Could not find 'demofinal.py'.")
    print("Make sure you renamed 'demofinal (1).py' to 'demofinal.py' and it is in this folder.")
    sys.exit()

def run_dataset_test(csv_filename):
    print(f"--- Loading dataset: {csv_filename} ---")
    
    df = None
    # Try different encodings to handle weird file formats
    encodings_to_try = ['utf-16', 'utf-8', 'latin1']
    
    for enc in encodings_to_try:
        try:
            # Try reading with the current encoding
            df = pd.read_csv(csv_filename, encoding=enc, on_bad_lines='skip', engine='python')
            
            # Clean up column names (remove weird invisible characters)
            df.columns = df.columns.str.strip().str.replace('\x00', '')
            
            # Check if this encoding gave us readable text
            if 'Sentence' in df.columns or 'Query' in df.columns:
                print(f"-> Successfully loaded using {enc} encoding.")
                break
        except Exception:
            continue
            
    if df is None:
        print("ERROR: Could not read the file with any standard encoding.")
        return

    # 2. DIAGNOSE COLUMNS
    print(f"File loaded. Found {len(df)} rows.")
    
    # Auto-detect the TEXT column
    text_col = None
    possible_text_names = ['Sentence', 'Query', 'text', 'sentence', 'payload', 'query']
    for name in possible_text_names:
        if name in df.columns:
            text_col = name
            break
            
    # Auto-detect the LABEL column
    label_col = None
    possible_label_names = ['Label', 'label', 'class', 'Class', 'is_attack']
    for name in possible_label_names:
        if name in df.columns:
            label_col = name
            break

    # Check if we found them
    if not text_col or not label_col:
        print(f"\nERROR: Columns look weird: {df.columns.tolist()}")
        print("Please check if the CSV file is corrupted.")
        return

    print(f"Using Text Column:  '{text_col}'")
    print(f"Using Label Column: '{label_col}'")
    print("-" * 30)

    # 3. RUN THE TEST
    correct_detections = 0
    false_positives = 0 
    false_negatives = 0 
    
    start_time = time.time()
    total_rows = len(df)

    print("Starting processing... (This will take a minute)")

    for index, row in df.iterrows():
        
        # --- PROGRESS BAR ---
        # Prints an update every 100 rows
        if index % 100 == 0:
            percent = (index / total_rows) * 100
            print(f"Processing row {index} of {total_rows} ({percent:.1f}%) ...")
        # --------------------

        # Get data
        text_input = str(row[text_col]) 
        
        try:
            actual_label = int(row[label_col])
        except (ValueError, TypeError):
            continue # Skip bad rows

        # Run your tool
        result = sql_injection_detection(text_input, verbose=False)
        
        # 1 = Block/Alert, 0 = Allow
        my_prediction = 1 if result['action'] in ["BLOCK", "ALERT"] else 0

        # Compare
        if my_prediction == actual_label:
            correct_detections += 1
        else:
            if my_prediction == 1 and actual_label == 0:
                false_positives += 1
            elif my_prediction == 0 and actual_label == 1:
                false_negatives += 1

    end_time = time.time()
    
    # 4. RESULTS
    if len(df) > 0:
        accuracy = (correct_detections / len(df)) * 100
    else:
        accuracy = 0

    print("\n" + "="*40)
    print(f"FINAL RESULTS")
    print("="*40)
    print(f"Rows Tested:          {len(df)}")
    print(f"Time Taken:           {end_time - start_time:.2f} seconds")
    print(f"Accuracy:             {accuracy:.2f}%")
    print("-" * 20)
    print(f"Correctly Identified: {correct_detections}")
    print(f"False Positives:      {false_positives} (Safe blocked)")
    print(f"False Negatives:      {false_negatives} (Attacks missed)")
    print("="*40)

if __name__ == "__main__":
    run_dataset_test("sql_injection_dataset.csv")