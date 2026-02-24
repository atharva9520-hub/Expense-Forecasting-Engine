from preprocess import preprocess_image
from ocr import extract_text
from parser import extract_total, extract_merchant_and_date
from classifier import categorize_receipt  
from pathlib import Path
import json

def run_pipeline(image_path: Path):
    print(f"  -> Preprocessing {image_path.name}...")
    processed_img = preprocess_image(image_path)

    print("  -> Running EasyOCR (English + Malay)...")
    text_results = extract_text(processed_img)

    print("  -> Running Document AI (LayoutLM)...")
    total_amount = extract_total(str(image_path))
    merchant_and_date = extract_merchant_and_date(str(image_path))

    print("  -> Running NLP Classification...")
    category, confidence = categorize_receipt(
        merchant_and_date.get("merchant"), 
        text_results
    )

    output = {
        "image_name": image_path.name,
        "image_path": str(image_path),
        "merchant": merchant_and_date.get("merchant"),
        "date": merchant_and_date.get("date"),
        "total_amount": total_amount,
        "category": category,              
        "category_confidence": confidence, 
        "raw_text_data": text_results 
    }

    return output

if __name__ == "__main__":
    folder_path = Path("/Users/atharvaaserkar/Documents/pp/financial_document_analysis/data/raw").resolve()
    image_paths = list(folder_path.glob("*.jpg"))

    if not image_paths:
        print(f"No images found in {folder_path}")
        exit(1)

    print(f"Found {len(image_paths)} images to process")
    print("=" * 60)

    output_file = Path("../data/extracted_receipts.json")

    # --- CHECKPOINTING SYSTEM ---
    # Load existing data so we don't start from scratch if the script was stopped
    if output_file.exists():
        with open(output_file, 'r') as f:
            try:
                all_results = json.load(f)
            except json.JSONDecodeError:
                all_results = [] # In case the file is empty or corrupted
                
        # Create a fast lookup set of images we already processed
        processed_images = {item['image_name'] for item in all_results}
        print(f"Resuming pipeline... Found {len(processed_images)} safely saved receipts.")
    else:
        all_results = []
        processed_images = set()

    # Loop through ALL images
    for i, image_path in enumerate(image_paths, 1):
        # Skip if already processed!
        if image_path.name in processed_images:
            print(f"[{i}/{len(image_paths)}] Skipping {image_path.name} (Already processed)")
            continue

        print(f"\n[{i}/{len(image_paths)}] Processing: {image_path.name}")
        print("-" * 60)

        try:
            result = run_pipeline(image_path)
            all_results.append(result)
            processed_images.add(image_path.name)

            print(f"    Success!")
            print(f"    Merchant: {result['merchant']}")
            print(f"    Date:     {result['date']}")
            print(f"    Total:    {result['total_amount']}")
            print(f"    Category: {result['category']}")

            # Checkpoint: Save the file immediately after every successful image
            with open(output_file, "w") as f:
                json.dump(all_results, f, indent=4)
            print(f" Checkpoint saved!")

        except Exception as e:
            print(f"  ✗ Error processing {image_path}: {e}")
