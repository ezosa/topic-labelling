import pickle
import numpy as np
import string
import re
import numpy as np
from nltk.corpus import stopwords
from gensim import corpora
from gensim.models import LdaModel
from sklearn.feature_extraction.text import CountVectorizer

from visualize import print_top_topic_words

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


def prune_vocabulary(documents, min_df=10, max_df=0.5, min_len=20):
    print("Truncating vocab with min_word_freq =", min_df, "and max_doc_prop =", max_df)
    docs = [" ".join(doc) for doc in documents]
    cvectorizer = CountVectorizer(min_df=min_df, max_df=max_df, stop_words=stops)
    cvectorizer.fit_transform(docs).sign()
    dictionary = list(cvectorizer.vocabulary_)
    print("Truncated vocab size:", len(dictionary))
    pruned_documents = []
    for doc in documents:
        pruned_doc = [w for w in doc if w in dictionary]
        if len(pruned_doc) >= min_len:
            pruned_documents.append(pruned_doc)
    return pruned_documents


def train_lda(articles, save_file, k=10, passes=500):
    common_dictionary = corpora.Dictionary(articles)
    common_corpus = [common_dictionary.doc2bow(d) for d in articles]
    corpus_filename = save_file + "_corpus.pkl"
    pickle.dump(common_corpus, open(corpus_filename, "wb"))
    print("Saved corpus at", corpus_filename)
    print("--- start LDA training ---")
    print("Train articles: ", str(len(articles)))
    print("Topics: ", k)
    print("Vocab len: ", len(common_dictionary))
    print("Save path:", save_file)
    print("Training LDA...")
    lda = LdaModel(corpus=common_corpus,
                   id2word=common_dictionary,
                   num_topics=k,
                   alpha='auto',
                   eta='auto',
                   passes=passes)
    lda.save(save_file)
    print("Done! Saved trained LDA model as", save_file, "!")
    print("LDA topics overview:")
    print_top_topic_words(lda)
    for t in range(k):
        print("Topic", str(t+1), ":", lda.print_topic(t, topn=20))
        print("-"*50)


def infer_document_topics(lda, docs, doc_ids, save_file, word_topics=False):
    print("Inferring topics for", len(docs), "docs")
    common_dictionary = lda.id2word
    corpus = [common_dictionary.doc2bow(doc) for doc in docs]
    # filter out documents composed entirely of OOV words
    #corpus = [c for c in corpus if len(c) > 0]
    indices = [i for i in range(len(corpus)) if len(corpus[i]) > 10]
    valid_corpus = [corpus[i] for i in indices]
    valid_ids = [doc_ids[i] for i in indices]
    topics_list = [lda.get_document_topics(valid_corpus[i], per_word_topics=word_topics) for i in range(len(valid_corpus))]
    topics_mat = convert_vectors_to_matrix(topics_list, lda.num_topics)
    print("Doc-topic matrix:", topics_mat.shape)
    print("Doc ids:", len(valid_ids))
    print("Sample ids:", valid_ids[:10])
    with open(save_file + ".npy", 'wb') as f:
        np.save(f, topics_mat)
        print("Done! Saved doc-topic matrix as", save_file + ".npy!")
    with open(save_file + "_ids.pkl", 'wb') as f:
        pickle.dump(valid_ids, f)
        f.close()
        print("Done! Saved doc ids as", save_file + "_ids.pkl!")
    return topics_mat


def convert_vectors_to_matrix(lda_vectors, n_topics=100):
    lda_mat = []
    for i, vec in enumerate(lda_vectors):
        vec_dict = dict(vec)
        array_vec = np.array([vec_dict[i] if i in vec_dict else 0.0 for i in range(n_topics)])
        lda_mat.append(array_vec)
    lda_mat = np.array(lda_mat)
    return lda_mat










