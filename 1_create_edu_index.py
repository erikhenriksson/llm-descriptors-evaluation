import json
import os
from tqdm import tqdm
from datasets import load_dataset
import argparse

def create_edu_index(output_path="edu_doc_ids.json", batch_size=10000, max_docs=None):
    """
    Create an index of document IDs from the FineWeb EDU dataset.
    
    Parameters:
    - output_path: Path to save the index
    - batch_size: Number of documents to process at once
    - max_docs: Maximum number of documents to index (for testing)
    
    Returns:
    - Path to the created index file
    """
    print(f"Loading FineWeb EDU dataset (streaming mode)...")
    
    # Stream the dataset to extract just the doc_ids
    edu_dataset = load_dataset("HuggingFaceFW/fineweb-edu", streaming=True)
    
    doc_ids = set()
    processed = 0
    
    print("Extracting document IDs...")
    try:
        for batch in tqdm(edu_dataset["train"].iter(batch_size=batch_size)):
            for doc in batch:
                doc_ids.add(doc["doc_id"])
                processed += 1
                
                # Periodically save progress
                if processed % 1000000 == 0:
                    print(f"Processed {processed} documents, found {len(doc_ids)} unique IDs")
                    # Save intermediate results
                    with open(f"{output_path}.partial", 'w') as f:
                        json.dump(list(doc_ids), f)
                
                if max_docs and processed >= max_docs:
                    break
            
            if max_docs and processed >= max_docs:
                break
    except Exception as e:
        print(f"Error during processing: {e}")
        print(f"Saving {len(doc_ids)} document IDs extracted so far...")
    
    print(f"Extraction complete. Found {len(doc_ids)} unique document IDs from {processed} documents.")
    
    # Save to file
    print(f"Saving index to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(list(doc_ids), f)
    
    print(f"Index saved successfully to {output_path}")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create an index of FineWeb EDU document IDs")
    parser.add_argument("--output", type=str, default="edu_doc_ids.json", 
                        help="Path to save the index file")
    parser.add_argument("--batch-size", type=int, default=10000, 
                        help="Number of documents to process at once")
    parser.add_argument("--max-docs", type=int, default=None, 
                        help="Maximum number of documents to process (for testing)")
    
    args = parser.parse_args()
    
    create_edu_index(
        output_path=args.output,
        batch_size=args.batch_size,
        max_docs=args.max_docs
    )