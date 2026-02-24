import sqlite3
import pandas as pd

def create_ground_truth_template():
    print("Connecting to database...")
    # Connect to your Gold Layer database
    conn = sqlite3.connect('/Users/atharvaaserkar/Documents/pp/financial_document_analysis/data/expenses.db')
    
    # Get all unique merchants and what your AI predicted for them
    query = """
    SELECT DISTINCT merchant, category AS predicted_category
    FROM receipts
    WHERE merchant IS NOT NULL AND merchant != 'UNKNOWN'
    """
    
    df = pd.read_sql_query(query, conn)
    
    # Add a blank column for you to type the actual correct category
    df['true_category'] = ""
    
    # Save it to a CSV file
    output_path = 'data/ground_truth_template.csv'
    df.to_csv(output_path, index=False)
    
    print(f"✅ Success! Template saved to {output_path}")
    print(f"👉 You have {len(df)} unique merchants to label.")
    
    conn.close()

if __name__ == "__main__":
    create_ground_truth_template()