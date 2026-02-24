import pandas as pd
from sklearn.metrics import classification_report, f1_score

def evaluate_zero_shot(csv_path):
    print("Loading classification ground truth data...")
    
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Could not find {csv_path}. Please check the file path.")
        return

    # IMPORTANT: Update these string names if your CSV columns are named differently!
    actual_col = 'actual_category' 
    predicted_col = 'predicted_category'

    if actual_col not in df.columns or predicted_col not in df.columns:
        print(f"Error: Make sure your CSV has both '{actual_col}' and '{predicted_col}' columns.")
        return

    # Drop any rows where you haven't manually entered the actual label yet
    df = df.dropna(subset=[actual_col, predicted_col])

    # Standardize the text (lowercase, strip whitespace) so "Groceries " matches "groceries"
    y_true = df[actual_col].astype(str).str.lower().str.strip()
    y_pred = df[predicted_col].astype(str).str.lower().str.strip()
    
    if len(y_true) == 0:
        print("Error: No labeled data found to evaluate!")
        return

    print("\n" + "="*55)
    print("ZERO-SHOT NLP CLASSIFICATION REPORT")
    print("="*55)
    
    # This scikit-learn function magically calculates Precision, Recall, and F1 for every category!
    report = classification_report(y_true, y_pred, zero_division=0)
    print(report)

    # Calculate the overall Macro F1-score for your resume
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    print("="*55)
    print(f"METRIC - Overall Macro F1-Score: {macro_f1:.2f}")
    print("="*55 + "\n")

if __name__ == "__main__":
    # Update this path to point to your actual CSV template
    csv_file = "/Users/atharvaaserkar/Documents/pp/financial_document_analysis/data/ground_truth_template.csv" 
    
    evaluate_zero_shot(csv_file)