import re # python regular expression library
import bs4 # python html parsing library "beautiful soup" 
import os
import gensim
import pandas as pd
DATA_DIR = 'data'

doc1 = """need to unlock a password protected Excel 2010 workbook <p>I have a user who had forgotten the password for an Excel 2010 file.&nbsp; She cannot open it.&nbsp; I tried changing the name to a zip file and opening the workbook file with an xml editor, but I can&#39;t get a readable format to come up so I can see the line of text with the password (so I can delete it).&nbsp; What I&#39;m getting is this gobbledy-gook:</p><p>\xc2\xba\xca\x9d\xc2\x99Z\xdc\xa1 \xc2\x84M/\xd8\xbd(+\xe8\x9d\xa4\xc2\xa47</p>"""

doc2 = """Password protected Excel 2016 spreadsheet---user forgot password! <p>Face palm time. \xc2\xa0A user has forgotten their password to an Excel 2016 spreadsheet.<br><br>Anybody have any slick tool that will at least let them open in read-only mode? \xc2\xa0Something to bust through the password would be excellent.</p>"""

doc3 = """How secure are password protected Excel files? <p>Once in a while if we need to send credentials to a third party we will use password protected Excel files sent via secure email (third party service where they have to login to see the email).<br><br></p><div>I\'m curious if the Excel file is secure enough by itself? Based on this info it looks like the default for Excel 2016 will be AES 256, which should be effectively secure at least against a brute force attack.<div><a href="https://technet.microsoft.com/en-us/library/cc179125%28v=office.16%29.aspx?f=255&amp;MSPPError=-2147217396">https://technet.microsoft.com/en-us/library/cc179125%28v=office.16%29.aspx?f=255&amp;MSPPError=-2147...</a><br></div><div><ul><li><i>"Lets you configure the CNG cipher algorithm that is used. </i><b><i>The default is AES</i></b><i>."</i></li><li><i>"Lets you configure the number of bits to use when you create the cipher key. </i><b><i>The default is 256 bits</i></b><i>."</i></li></ul></div>\xc2\xa0</div>"""
# Target Corpus

docs = [doc1, doc2, doc3]

# Helper IO Function
def list2txt(doc_list, filename):
    filepath = os.path.join(DATA_DIR, filename)
    outfile = open(filepath, 'w')
    outfile.write("\n".join(doc_list))
    return filepath

default_target_corpus = gensim.corpora.TextCorpus(list2txt(docs, 'raw.txt'))
default_target_corpus.dictionary.items()

def print_default_preprocessing(n):
    default_target_list = [doc for doc in default_target_corpus.get_texts()]
    print ' '.join(default_target_list[n-1])

###################################################################
# Sample custom preprocessing sequence
###################################################################
DELIMITER_PATTERN = u'[!?,;:\t\\\\"\\(\\)\\\'\u2026\u201c\u2013\u2019\u2026\n]|\\s\\-\\s|\.\s'
TOKEN_PATTERN = r'(?u)[\_][a-zA-Z0-9\_]*|[a-zA-Z0-9][a-zA-Z0-9.]*\b'

def html_parser(html):
    try:
        html = re.sub(r"<img.*?>", " __img__ ", html)
        html = re.sub(r"<a .*?/a>", " __url__ ", html)
        soup = bs4.BeautifulSoup(html, "html.parser")
        for br in soup.find_all("br"):
            br.replace_with("\n")
        return soup.get_text()
    except:  
        return ''
    
def split_sentences(doc):
    try:
        delimiters = re.compile(DELIMITER_PATTERN)
        sentences = delimiters.split(doc)
        return sentences
    except:
        return []

def tokenizer(sentence, token_pattern=TOKEN_PATTERN, lowercase=True):
    try:
        token_pattern = re.compile(token_pattern)
        if lowercase:
            sentence = sentence.lower()
        return token_pattern.findall(sentence)

    except:
        return []

def clean_text(html, lowercase=True):
    text = html_parser(html)
    if lowercase:
        text = text.lower()

    sents = split_sentences(text)
    sents = [' '.join(tokenizer(sent, lowercase=lowercase)) for sent in sents]
    try:
        sents = map(lambda x: x.strip(), sents)
    except:
        pass
    try:
        sents = [sent for sent in sents if len(sent) > 0]
    except:
        pass
    return ' '.join(sents)

def print_custom_preprocessing(doc):
    print clean_text(doc)
    
# Subclass gensim TextCorpus object to apply custom preprocesing
# Preprocessing functions need to be efficient if performance is a concern!
class CustomTextCorpus(gensim.corpora.TextCorpus):
  def get_texts(self):
    for doc in self.getstream():
        yield [word for word in clean_text(doc).split()]
  def __len__(self): 
    self.length = sum(1 for _ in self.get_texts())
    return self.length

preprocessed_docs = [clean_text(doc) for doc in docs]
custom_target_corpus = CustomTextCorpus(list2txt(preprocessed_docs, 'preprocessed.txt'))
custom_target_corpus.dictionary.items()


# Make Pandas dataframe preview font larger
# Set CSS properties for th elements in dataframe
# Set CSS properties for td elements in dataframe
# Set table display styles
styles = [
  dict(selector="th", props=[('font-size', '30px')]),
  dict(selector="td", props=[('font-size','30px')])
  ]

def get_wordcount_df(n, prepro="default"):
    if prepro == "custom":
        target_corpus = custom_target_corpus
    else: 
        target_corpus = default_target_corpus
        
    doc_vectors = [word_count_vector for word_count_vector in target_corpus]
    
    return pd.DataFrame([(k,target_corpus.dictionary.id2token[k],v) for k,v in doc_vectors[n-1]], 
                            columns=['vocab_index','token','word_count']).set_index('vocab_index')    
    
def display_wordcount_vector(n, prepro="default"):
    return get_wordcount_df(n, prepro).head(20).style.set_table_styles(styles)


# IDF's coming from the small target corpus
def get_target_tfidf(target_corpus):
    return gensim.models.TfidfModel(target_corpus, dictionary=target_corpus.dictionary)[target_corpus]

iter(get_target_tfidf(default_target_corpus)).next()

# Term mapping logic
# Subclass the corpus transormation interface
class MapTermIds(gensim.interfaces.TransformationABC):
  def __init__(self, termid_map):
      self.termid_map = termid_map

  def __getitem__(self, bow):
      is_corpus, bow = gensim.utils.is_corpus(bow)
      if is_corpus:
          return self._apply(bow)

      vector = [(self.termid_map.get(termid), weight)
                for termid, weight in bow if self.termid_map.get(termid)]
      return vector
    
def create_compatible_termid_maps(target_dictionary, training_dictionary):
    token_training_map = training_dictionary.token2id
    target_training_map = {k: token_training_map.get(v) for k, v
                       in target_dictionary.iteritems()}
    missing_indices = [k for k, v in target_training_map.iteritems() if not v]
    target_dictionary.filter_tokens(missing_indices)
    
    # Remove terms not found in the training dictionary
    target_dictionary.compactify() 

    target_training_map = {k: token_training_map.get(v) for k, v
                       in target_dictionary.iteritems()}
    training_target_map = {v: k for k, v in target_training_map.iteritems()}
    target_training_transform = MapTermIds(target_training_map)
    training_target_transform = MapTermIds(training_target_map)

    return target_dictionary, target_training_transform, training_target_transform

# Uses custom method "create_compatible_termid_maps"
# IDF's coming from the large training corpus
training_tfidf = gensim.models.TfidfModel.load('models/posts_parsed_text.tfidf')
def get_training_tfidf(target_corpus):
    training_dictionary = gensim.corpora.Dictionary.load(
        os.path.join(DATA_DIR, 'posts_parsed_text.dict'))
    target_dictionary, target_training_transform, training_target_transform = \
        create_compatible_termid_maps(target_corpus.dictionary, training_dictionary)
    return target_dictionary, training_target_transform[training_tfidf[
        target_training_transform[target_corpus]]]

custom_target_dictionary, custom_background_tfidf = get_training_tfidf(custom_target_corpus)
default_target_dictionary, default_background_tfidf = get_training_tfidf(default_target_corpus)

def get_tfidf_df(n, prepro='default'):
    if prepro=='custom':
        background_tfidf = custom_background_tfidf
        target_dictionary = custom_target_dictionary
    else: 
        background_tfidf = default_background_tfidf
        target_dictionary = default_target_dictionary
    
    tfidf_vectors = [vec for vec in background_tfidf]
    wordcount_df = get_wordcount_df(n, prepro)
    tfidf_df = pd.DataFrame(
        [(k,v) for k,v in tfidf_vectors[n-1]], 
        columns=['vocab_index', 'tfidf_score']).set_index('vocab_index')
    df = wordcount_df.join(tfidf_df)
    return df

def display_tfidf_vector(n, prepro='default'):
    return get_tfidf_df(n, prepro).head(20).style.set_table_styles(styles)

# Display a doc-similarity matrix as a pandas dataframe indexed by the terms 
def display_doc_similarities(M):
    terms = ['doc1', 'doc2', 'doc3']
    return pd.DataFrame(M, index=terms, columns=terms).style.set_table_styles(styles)

cosim_index = gensim.similarities.MatrixSimilarity(custom_target_corpus)
cosim_index_tfidf = gensim.similarities.MatrixSimilarity(custom_background_tfidf)

# skip-gram, min_count=20, size=200, window=10
w2v = gensim.models.Word2Vec.load("models/word2vec_model_unigram_lowercase")
ftx = gensim.models.FastText.load("models/fasttext_model_unigram_lowercase")

# Display a term-similarity matrix as a pandas dataframe indexed by the terms 
def display_term_similarities(term_dict, M):
    terms = [term_dict[k] for k in sorted(term_dict.iterkeys())]
    return pd.DataFrame(M.todense(), 
             index=terms, 
             columns=terms).style.set_table_styles(styles)

dict_stub = {0: "broken", 1: "fix", 2:"laptop", 3:"computer", 4:"crashed", 5:"troubleshoot"}
M_eg = w2v.wv.similarity_matrix(dict_stub)

target_corpus = custom_target_corpus
M_w2v = w2v.wv.similarity_matrix(target_corpus.dictionary)
M_ftx = ftx.wv.similarity_matrix(target_corpus.dictionary)
#display_term_similarities(target_corpus.dictionary, M_ftx)

doc_w2vsim_index = gensim.similarities.SoftCosineSimilarity(custom_background_tfidf, M_w2v)
doc_ftxsim_index = gensim.similarities.SoftCosineSimilarity(custom_background_tfidf, M_ftx)