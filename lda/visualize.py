import pickle
import pyLDAvis
import pyLDAvis.gensim
from gensim.models import LdaModel


def write_top_topic_words_to_file(model_path, save_filename, n_words=20):
    print("Loading model from", model_path)
    model = LdaModel.load(model_path)
    dictionary = model.id2word
    print("Vocab size:", len(dictionary))
    top_words = []
    with open(save_filename, 'w') as f:
        for k in range(model.num_topics):
            topic_words = [w[0] for w in model.show_topic(k, topn=n_words)]
            top_words.append(topic_words)
            print("Topic", k+1, ":", ", ".join(topic_words))
            f.write(" ".join(topic_words) + "\n")
        f.close()


def print_top_topic_words(model, n_words=20):
    #print("Loading model from", model_path)
    #lda = LdaModel.load(model_path)
    dictionary = model.id2word
    print("Vocab size:", len(dictionary))
    top_words = []
    for k in range(model.num_topics):
        topic_words = [w[0] for w in model.show_topic(k, topn=n_words)]
        top_words.append(topic_words)
        print("Topic", k+1, ":", ", ".join(topic_words))


def get_top_words_weights(model, n_words=20):
    #print("Loading model from", model_path)
    #lda = LdaModel.load(model_path)
    top_words = []
    for k in range(model.num_topics):
        #top_words = [w[0] for w in lda.show_topic(k, topn=n_words)]
        top_words.append(model.show_topic(k, topn=n_words))
    return top_words


def get_top_words(model, n_words=20):
    top_words = []
    for k in range(model.num_topics):
        topic_words = [w[0] for w in model.show_topic(k, topn=n_words)]
        top_words.append(topic_words)
    return top_words


def create_pyldavis_plot(model_path, model_name):
    print("Loading model from", model_path)
    lda = LdaModel.load(model_path + model_name)
    common_dictionary = lda.id2word
    common_corpus = pickle.load(open(model_path + model_name + "_corpus.pkl", "rb"))
    vis_data = pyLDAvis.gensim.prepare(lda, common_corpus, common_dictionary, sort_topics=False)
    #model_name = model_file.split("/")[-1]
    out_filename = model_path + model_name + "_pyldavis.html"
    outfile = open(out_filename, 'w')
    pyLDAvis.save_html(vis_data, fileobj=outfile)
    print("Done! Saved PyLDAVis plot as", out_filename)

