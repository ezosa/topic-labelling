import string
import json
import re
import os
import numpy as np
from random import shuffle
from nltk.corpus import stopwords
from spacy.lang.en.stop_words import STOP_WORDS
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

exclude = set(string.punctuation)

stops = {'fi': set(stopwords.words('finnish')),
         'fr': set(stopwords.words('french')),
         'de': set(stopwords.words('german')),
         'en': set(stopwords.words('english') + ['links', 'referencesexternal'])
         }


def clean_document(doc, lang='en'):
    clean_punc = ''.join(ch if ch not in exclude else ' ' for ch in doc.lower())
    clean_punc_tokens = clean_punc.split()
    clean_stop = [tok for tok in clean_punc_tokens if tok not in stops[lang]]
    clean_digits = [tok for tok in clean_stop if re.match(r'^([\s\d]+)$', tok) is None]
    clean_short = [tok for tok in clean_digits if 2 < len(tok) < 20]
    # if lang == 'en':
    #     clean_short = [lemmatizer.lemmatize(tok) for tok in clean_digits]
    # else:
    #     clean_short = [tok for tok in clean_digits if tok not in stops and len(tok) > 2]
    return clean_short


def parse_wikipedia_dump():
    filepath = "/proj/zosa/newseye_data/wikipedia/"
    filename = "enwiki-20181001-corpus.txt"
    print('Opening Wikipedia dump at', filepath+filename)
    text = open(filepath+filename, 'r', encoding='utf-8').read()
    print("Parsing Wiki dump")
    articles = text.split("</article>")
    article_dict = {}
    for i, article in enumerate(articles):
        if i == 0:
            start_index = 0
        else:
            start_index = 1
        article_lines = article.split("\n")
        article_title = article_lines[start_index][15:len(article_lines[start_index])-2]
        article_text = "".join(article_lines[start_index+1:])
        article_dict[article_title] = article_text
    print("Done parsing! Articles:", len(article_dict))
    out_filename = "data/wiki/" + filename[:-4]+'.json'
    with open(out_filename, 'w') as json_file:
        json.dump(article_dict, json_file)
        json_file.close()
        print("Saved parsed articles to", out_filename)


def compute_tfidf_for_corpus(texts):
    print("Computing TF-IDF scores for corpus of size", len(texts))
    texts = [" ".join(text) for text in texts]
    vectorizer = TfidfVectorizer(analyzer='word', stop_words='english', min_df=10, max_df=0.8)
    tfidf_values = vectorizer.fit_transform(texts)
    vocab = vectorizer.get_feature_names()
    tfidf_sums = np.sum(tfidf_values, axis=0)
    print("tfidf_sums:", tfidf_sums.shape)
    tfidf_dict = {vocab[i]: tfidf_sums[0, i] for i in range(len(vocab))}
    return tfidf_dict


def prepare_wiki_train_data(json_file, n_top_terms=30, train_size=500000, valid_size=10000, test_size=10000):
    print("Loading Wikipedia JSON dump from", json_file)
    data = json.load(open(json_file, 'r'))
    article_titles = list(data.keys())
    shuffle(article_titles)
    max_articles = 600000
    print("Articles:", len(data))
    article_texts = [clean_document(data[article_titles[index]]) for index in range(max_articles)]
    source_target_tuples = []
    #if top_terms_strategy == 'tfidf':
    tfidf_dict = compute_tfidf_for_corpus(article_texts)
    for index in range(max_articles):
        article_text = article_texts[index]
        if len(article_titles[index]) > 2 and len(article_text) > n_top_terms:
            article_words = list(set(article_text))
            #if top_terms_strategy == 'sent':
            top_words_sent = ' '.join(article_words[:n_top_terms])
            #else:
            words_tfidf = [(word, tfidf_dict[word]) for word in article_words if word in tfidf_dict]
            top_words_tfidf = sorted(words_tfidf, key=lambda tup: tup[1], reverse=True)
            top_words_tfidf = [word[0] for word in top_words_tfidf][:n_top_terms]
            top_words_tfidf = ' '.join(top_words_tfidf)
            if index % 2000 == 0:
                print("Title:", article_titles[index])
                print("Top words sent:", top_words_sent)
                print("Top words tfidf:", top_words_tfidf)
            source_target_tuples.append((article_titles[index].lower(), top_words_sent, top_words_tfidf))
    print("source-target pairs:", len(source_target_tuples))
    #shuffle(source_target_tuples)
    train_target = [source_target_tuples[i][0] for i in range(train_size)]
    train_source_sent = [source_target_tuples[i][1] for i in range(train_size)]
    train_source_tfidf = [source_target_tuples[i][2] for i in range(train_size)]

    valid_target = [source_target_tuples[i][0] for i in range(train_size, train_size+valid_size)]
    valid_source_sent = [source_target_tuples[i][1] for i in range(train_size, train_size+valid_size)]
    valid_source_tfidf = [source_target_tuples[i][2] for i in range(train_size, train_size+valid_size)]

    test_target = [source_target_tuples[i][0] for i in range(train_size+valid_size, train_size+valid_size+test_size)]
    test_source_sent = [source_target_tuples[i][1] for i in range(train_size+valid_size, train_size+valid_size+test_size)]
    test_source_tfidf = [source_target_tuples[i][2] for i in range(train_size+valid_size, train_size+valid_size+test_size)]

    out_filepath = 'results/topic-labelling/wiki/'
    # write train set
    with open(os.path.join(out_filepath, 'train_sent.source'), 'a') as f:
        f.write('\n'.join(train_source_sent) + '\n')
        f.close()
    with open(os.path.join(out_filepath, 'train_sent.target'), 'a') as f:
        f.write('\n'.join(train_target) + '\n')
        f.close()
    with open(os.path.join(out_filepath, 'train_tfidf.source'), 'a') as f:
        f.write('\n'.join(train_source_tfidf) + '\n')
        f.close()
    with open(os.path.join(out_filepath, 'train_tfidf.target'), 'a') as f:
        f.write('\n'.join(train_target) + '\n')
        f.close()

    # write valid set
    with open(os.path.join(out_filepath, 'valid_sent.source'), 'a') as f:
        f.write('\n'.join(valid_source_sent) + '\n')
        f.close()
    with open(os.path.join(out_filepath, 'valid_sent.target'), 'a') as f:
        f.write('\n'.join(valid_target) + '\n')
        f.close()
    with open(os.path.join(out_filepath, 'valid_tfidf.source'), 'a') as f:
        f.write('\n'.join(valid_source_tfidf) + '\n')
        f.close()
    with open(os.path.join(out_filepath, 'valid_tfidf.target'), 'a') as f:
        f.write('\n'.join(valid_target) + '\n')
        f.close()

    # write test set
    with open(os.path.join(out_filepath, 'test_sent.source'), 'a') as f:
        f.write('\n'.join(test_source_sent) + '\n')
        f.close()
    with open(os.path.join(out_filepath, 'test_sent.target'), 'a') as f:
        f.write('\n'.join(test_target) + '\n')
        f.close()
    with open(os.path.join(out_filepath, 'test_tfidf.source'), 'a') as f:
        f.write('\n'.join(test_source_tfidf) + '\n')
        f.close()
    with open(os.path.join(out_filepath, 'test_tfidf.target'), 'a') as f:
        f.write('\n'.join(test_target) + '\n')
        f.close()


json_file = "/proj/zosa/data/wiki/enwiki-20181001-corpus.json"
prepare_wiki_train_data(json_file=json_file)