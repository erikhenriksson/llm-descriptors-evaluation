import argparse
import json
import torch
from torch.nn.functional import sigmoid
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict web registers for documents in a JSONL file"
    )
    parser.add_argument(
        "--input_file", type=str, required=True, help="Path to input JSONL file"
    )
    parser.add_argument(
        "--output_file", type=str, required=True, help="Path to output JSONL file"
    )
    parser.add_argument(
        "--base_model", type=str, default="xlm-roberta-large", help="Base model name"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="TurkuNLP/web-register-classification-en",
        help="Model name",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for processing"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    return parser.parse_args()


def read_jsonl(file_path):
    """Read JSONL file line by line"""
    documents = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            documents.append(json.loads(line))
    return documents


def write_jsonl(documents, file_path):
    """Write documents to JSONL file"""
    with open(file_path, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")


def get_batches(documents, batch_size):
    """Split documents into batches"""
    for i in range(0, len(documents), batch_size):
        yield documents[i : i + batch_size]


def predict_registers(documents, model, tokenizer, device, batch_size):
    """Predict registers for documents and add probabilities to each document"""
    model.to(device)
    model.eval()
    id2label = model.config.id2label

    for batch in tqdm(
        get_batches(documents, batch_size),
        total=(len(documents) + batch_size - 1) // batch_size,
    ):
        texts = [doc["document"] for doc in batch]

        # Tokenize batch
        inputs = tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        ).to(device)

        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            # Apply sigmoid for multi-label classification
            probs = sigmoid(logits).cpu().numpy()

        # Add predictions to documents
        for i, doc in enumerate(batch):
            register_probs = {
                id2label[j]: round(float(probs[i][j]), 2) for j in range(len(id2label))
            }
            doc["register_probabilities"] = register_probs


def main():
    args = parse_args()

    # Load tokenizer and model
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)

    # Read input JSONL
    print(f"Reading documents from: {args.input_file}")
    documents = read_jsonl(args.input_file)
    print(f"Found {len(documents)} documents")

    # Predict registers
    print("Predicting registers...")
    predict_registers(documents, model, tokenizer, args.device, args.batch_size)

    # Write output JSONL
    print(f"Writing results to: {args.output_file}")
    write_jsonl(documents, args.output_file)
    print("Done!")


if __name__ == "__main__":
    main()
