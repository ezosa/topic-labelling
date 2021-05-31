import pandas as pd
from random import shuffle

from train_lda import train_lda, clean_document, prune_vocabulary


import argparse
argparser = argparse.ArgumentParser()
argparser.add_argument('--csv_file', default='yle_2018.csv', type=str, help="csv file containing df")
argparser.add_argument('--num_topics', default=100, type=int, help="no. of topics to train")
argparser.add_argument('--min_len', default=20, type=int, help="min. article length")
argparser.add_argument('--max_len', default=300, type=int, help="max. article length")
argparser.add_argument('--min_df', default=10, type=int, help="min doc freq of tokens")
argparser.add_argument('--max_df', default=0.5, type=int, help="max doc freq of tokens")
argparser.add_argument('--max_art', default=15000, type=int, help="max. no. of articles in training data")
argparser.add_argument('--save_path', default='yle/', type=str, help="save path for trained LDA")
args = argparser.parse_args()

print("\n\n" + "="*10, "Train LDA on Yle data", "="*10)
print("csv_file:", args.csv_file)
print("num_topics:", args.num_topics)
print("min_len:", args.min_len)
print("max_len:", args.max_len)
print("max_art:", args.max_art)
print("save_path:", args.save_path)
print("="*50 + "\n\n")

print("Loading dataframe from", args.csv_file)

df = pd.read_csv(args.csv_file)
#print("df.shape:", df.shape)

articles = list(df.content)
articles = [clean_document(doc) for doc in articles]
articles = [doc for doc in articles if args.min_len < len(doc) < args.max_len]

print("Articles:", len(articles))
shuffle(articles)
articles = articles[:args.max_art]
articles = prune_vocabulary(articles, min_df=args.min_df, max_df=args.max_df, min_len=args.min_len)

labels = list(df.subjects)
labels = [l.split(',') for l in labels]

model_name = args.csv_file.split("/")[-1][:-4] + "_lda_"+str(args.num_topics)+"topics"
save_file = args.save_path + model_name
print("save path:", save_file)

# train LDA model
train_lda(articles, save_file, k=args.num_topics, passes=100)