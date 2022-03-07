from sentence_transformers import SentenceTransformer
import pandas as pd
import time
import os
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--csv_file', default='', type=str)
argparser.add_argument('--data_path', default='', type=str)
argparser.add_argument('--sbert_model', default='distiluse-base-multilingual-cased', type=str)
args = argparser.parse_args()

print("\n" + "-"*5, "Encode articles using SBERT", "-"*5)
print("csv_file:", args.csv_file)
print("data_path:", args.data_path)
print("sbert_model:", args.data_path)
print("-"*30 + "\n\n")


def encode(df, save_filepath):
    try:
        documents = list(df.content)
        model_name = args.sbert_model
        model = SentenceTransformer(model_name)
        print(f"[!] Encoding", len(documents), "articles")

        now = time.time()
        enc = model.encode(documents)
        encdf = pd.DataFrame(enc)
        if 'tags' in df.columns:
            tags = list(df.tags)
            encdf['tags'] = tags

        print(f"[!] df shape: {df.shape}")
        print(f"[!] Took {time.time() - now}s")
        encdf.to_csv(save_filepath, index=False)
        print(f"[+] Written to {save_filepath}")

    except Exception as ex:
        print(ex)


if __name__ == "__main__":
    articles_filepath = os.path.join(args.data_path, args.csv_file)
    df = pd.read_csv(articles_filepath)
    save_filepath = articles_filepath[:-4]+'_sbert.csv'
    encode(df, save_filepath)

