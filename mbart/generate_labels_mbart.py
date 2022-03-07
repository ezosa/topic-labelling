import os
import torch
from transformers import MBartTokenizer, MBartForConditionalGeneration


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:", device)

import argparse
argparser = argparse.ArgumentParser()
argparser.add_argument('--model_name', default='mbart-', type=str, help="model name")
argparser.add_argument('--save_path', default='trained_models/', type=str, help="save directory")
argparser.add_argument('--topics_file', default='topics.txt', type=str, help="")
argparser.add_argument('--num_labels', default=5, type=int, help="num of labels to generate")
argparser.add_argument('--min_length', default=1, type=int, help="min label length")
argparser.add_argument('--max_length', default=5, type=int, help="max label length")
args = argparser.parse_args()

print("\n"+"-"*10, "Generate topic labels from finetuned mBART", "-"*10)
print("model_name:", args.model_name)
print("save_path:", args.save_path)
print("topics_file:", args.topics_file)
print("num_labels:", args.num_labels)
print("min_length:", args.min_length)
print("max_length:", args.max_length)
print("-"*70 + "\n")


print("Loading tokenizer")
tokenizer = MBartTokenizer.from_pretrained(
    "facebook/mbart-large-cc25", src_lang="fi_FI", tgt_lang="fi_FI")

print("Loading finetuned model")
MODEL_OUT_DIR = os.path.join(args.save_path, args.model_name + ".pt")
model = MBartForConditionalGeneration.from_pretrained(MODEL_OUT_DIR)

model.config.decoder_start_token_id = tokenizer.lang_code_to_id["fi_FI"]
model.eval()

print("Loading topics")
topics = open(args.topics_file, 'r', encoding='utf-8').readlines()
topics = [t.strip() for t in topics]
print("Topics:", len(topics))

# encode topics with tokenizer
inputs = tokenizer(topics,
                   padding="max_length",
                   max_length=32,
                   truncation=True,
                   return_tensors="pt")

# generate labels with encoded tokens
print("Generating labels for", len(topics), "topics")
output_tokens = model.generate(**inputs,
                               decoder_start_token_id=tokenizer.lang_code_to_id["fi_FI"],
                               num_return_sequences=args.num_labels,
                               num_beams=args.num_labels,
                               min_length=args.min_length,
                               max_length=args.max_length)

# decode model outputs
generated_labels = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)

final_labels = []
for k in range(len(topics)):
    start_index = k*args.num_labels
    topic_labels = generated_labels[start_index:start_index + args.num_labels]
    print("-"*30)
    print("Topic", k+1, ":", topics[k]+"\n")
    for i, label in enumerate(topic_labels):
        print("Label", i+1, ":", label)
    final_labels.append(", ".join(topic_labels))

# write outputs to file
output_file = MODEL_OUT_DIR[:-3] + "_output.txt"
final_out_path = os.path.join(output_file)
with open(final_out_path, 'w', encoding='utf-8') as f:
    for k in range(len(final_labels)):
        f.write(final_labels[k]+"\n")

print("Done! Saved final labels to", final_out_path)