import json
import os
import string
import re
import numpy as np
import pandas as pd
from random import shuffle
from nltk.corpus import stopwords
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

exclude = set(string.punctuation)

finnish_stops_file = "/users/zosaelai/project_dir/datasets/stopwords_finnish.txt"
finnish_stops = open(finnish_stops_file, 'r', encoding='utf-8').readlines()
finnish_stops = [s.split()[1].lower().strip() for s in finnish_stops]
finnish_stops = finnish_stops + ['yle', 'suomen', 'uutiset', 'sanoo', 'http', 'https', 'hyvin',
                                 'sitten', 'pitää', 'viime', 'www', 'vain', 'vuotta', 'html',
                                 'com', 'yksi', 'kaksi', 'kolme', 'neljä', 'viisi', 'kuusi',
                                 'seitsemän', 'kahdeksan', 'ydehksän', 'kymmenen', 'ssä', 'ylen',
                                 'vuonna', 'kertoo', 'suomessa', 'ajan', 'nykyään', 'hyvä', 'tuli',
                                 'vuoden', 'tuli', 'tänään', 'nousi', 'een']
stops = {'fi': set(stopwords.words('finnish') + finnish_stops),
         'fr': set(stopwords.words('french')),
         'de': set(stopwords.words('german')),
         'en': set(stopwords.words('english'))
         }


def clean_document(doc, lang='fi'):
    clean_punc = ''.join(ch if ch not in exclude else ' ' for ch in doc.lower())
    clean_punc_tokens = clean_punc.split()
    clean_stop = [tok for tok in clean_punc_tokens if tok not in stops[lang]]
    clean_digits = [tok for tok in clean_stop if re.match(r'^([\s\d]+)$', tok) is None]
    clean_short = [tok for tok in clean_digits if 2 < len(tok) < 20]
    return clean_short


def get_articles_dataframe(parent_dir, lang='fi', start_year=2017, end_year=2017):
    print("Processing articles from", parent_dir, "-", start_year, "to", end_year)
    article_dict = {'id': [], 'content': [], 'headline': [], 'subjects': []}
    for year in range(start_year, end_year+1):
        year_path = os.path.join(parent_dir, str(year))
        months = sorted(os.listdir(year_path))
        for month in months:
            month_path = os.path.join(year_path, month)
            json_files = sorted(os.listdir(month_path))
            for json_file in json_files:
                json_filepath = os.path.join(month_path, json_file)
                data = json.load(open(json_filepath, 'r'))['data']
                for article in data:
                    if 'subjects' in article:
                        art_id = article['id']
                        art_content = ''
                        art_headline = ''
                        for content in article['content']:
                            if content['type'] == 'text':
                                art_content += ' ' + content['text']
                            elif content['type'] == 'heading':
                                art_headline = content['text']
                        art_subjects = []
                        for subject in article['subjects']:
                            if 'title' in subject and lang in subject['title']:
                                subj_name = subject['title'][lang]
                                #subj_id = subject['id']
                                #subj_dict = {'id': subj_id, 'name': subj_name}
                                if subj_name is not None:
                                    art_subjects.append(subj_name)
                        #print("subjects:", art_subjects)
                        article_dict['id'].append(art_id)
                        article_dict['content'].append(art_content)
                        article_dict['headline'].append(art_headline)
                        article_dict['subjects'].append(','.join(art_subjects))
    print("Done processing articles!")
    print("Articles:", len(article_dict['id']))
    article_df = pd.DataFrame.from_dict(article_dict)
    article_df = article_df.dropna()
    return article_df


def compute_tfidf_for_corpus(texts):
    print("Computing TF-IDF scores for corpus of size", len(texts))
    texts = [" ".join(text) for text in texts]
    vectorizer = TfidfVectorizer(analyzer='word', min_df=10, max_df=0.25)
    tfidf_values = vectorizer.fit_transform(texts)
    vocab = vectorizer.get_feature_names()
    print("vocab:", len(vocab))
    tfidf_sums = np.sum(tfidf_values, axis=0)
    print("tfidf_sums:", tfidf_sums.shape)
    tfidf_dict = {vocab[i]: tfidf_sums[0, i] for i in range(len(vocab))}
    return tfidf_dict


def prepare_yle_train_data(csv_file, save_path, n_top_terms=30, top_words_strategy='tfidf'):
    print("Loading df from:", csv_file)
    print("save_path:", save_path)
    print("top_words_strategy:", top_words_strategy)
    df = pd.read_csv(csv_file)
    article_texts = list(df.content)
    article_texts = [clean_document(doc) for doc in article_texts]
    labels = list(df.subjects)
    labels = [s.split(",") for s in labels]
    # exclude labels that are only used once
    label_counts = Counter([l for sublist in labels for l in sublist])
    source_target_tuples = []
    if top_words_strategy == 'tfidf':
        tfidf_dict = compute_tfidf_for_corpus(article_texts)
    for index in range(len(article_texts)):
        article_text = article_texts[index]
        article_labels = [l for l in labels[index] if label_counts[l] > 1]
        if len(article_labels) > 0 and len(article_text) > n_top_terms:
            if top_words_strategy == 'tfidf':
                article_words = list(set(article_text))
                words_tfidf = [(word, tfidf_dict[word]) for word in article_words if word in tfidf_dict]
                top_words_tfidf = sorted(words_tfidf, key=lambda tup: tup[1], reverse=True)
                top_words = [word[0] for word in top_words_tfidf][:n_top_terms]
            else:
                top_words = article_text[:n_top_terms]
            top_words = ' '.join(top_words)
            if index % 5000 == 0:
                print("Labels:", article_labels)
                print("Top words:", top_words)
            for art_label in article_labels:
                source_target_tuples.append((art_label.lower(), top_words))
    print("source-target pairs:", len(source_target_tuples))
    shuffle(source_target_tuples)
    total_size = len(source_target_tuples)
    train_size = int(0.8 * total_size)
    valid_size = int(0.1 * total_size)
    test_size = int(0.1 * total_size)

    train_target = [source_target_tuples[i][0] for i in range(train_size)]
    train_source_tfidf = [source_target_tuples[i][1] for i in range(train_size)]

    valid_target = [source_target_tuples[i][0] for i in range(train_size, train_size+valid_size)]
    valid_source_tfidf = [source_target_tuples[i][1] for i in range(train_size, train_size+valid_size)]

    test_target = [source_target_tuples[i][0] for i in range(train_size+valid_size, train_size+valid_size+test_size)]
    test_source_tfidf = [source_target_tuples[i][1] for i in range(train_size+valid_size, train_size+valid_size+test_size)]

    out_filepath = save_path
    # write train set
    with open(os.path.join(out_filepath, 'train_'+top_words_strategy+'.source'), 'a') as f:
        f.write('\n'.join(train_source_tfidf) + '\n')
        f.close()
    with open(os.path.join(out_filepath, 'train_'+top_words_strategy+'.target'), 'a') as f:
        f.write('\n'.join(train_target) + '\n')
        f.close()

    # write valid set
    with open(os.path.join(out_filepath, 'valid_'+top_words_strategy+'.source'), 'a') as f:
        f.write('\n'.join(valid_source_tfidf) + '\n')
        f.close()
    with open(os.path.join(out_filepath, 'valid_'+top_words_strategy+'.target'), 'a') as f:
        f.write('\n'.join(valid_target) + '\n')
        f.close()

    # write test set
    with open(os.path.join(out_filepath, 'test_'+top_words_strategy+'.source'), 'a') as f:
        f.write('\n'.join(test_source_tfidf) + '\n')
        f.close()
    with open(os.path.join(out_filepath, 'test_'+top_words_strategy+'.target'), 'a') as f:
        f.write('\n'.join(test_target) + '\n')
        f.close()
    print("Done! Save datasets to", out_filepath, "!")


def convert_csv_to_text(csv_file, save_path):
    df = pd.read_csv(csv_file)
    art_ids = list(df.id)
    art_text = list(df.content)
    for i in range(len(art_ids)):
        with open(os.path.join(save_path, art_ids[i]+'.txt'), 'w') as f:
            f.write("ArticleID:" + art_ids[i] + "\n" + art_text[i])
            f.close()
    print("Done! Saved text files", save_path, "!")



def main():
    yle_path = "project_dir/elaine/topic-labelling/yle/txt_files/"
    df_file = "yle_2017.csv"
    convert_csv_to_text(df_file, yle_path)
    # prepare_yle_train_data(csv_file=yle_path+df_file, save_path=yle_path, top_words_strategy='sent')
    # out_filename = "results/topic-labelling/yle_"+str(args.start_year)+".csv"
    # print("save filename:", out_filename)
    # df = get_articles_dataframe(parent_dir=args.yle_path, start_year=args.start_year, end_year=args.end_year)
    # df.to_csv(out_filename, index=False)
    # print("Done! Saved df to", out_filename, "!")


if __name__ == '__main__':
    main()



