from pypdf import PdfReader
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import matplotlib.pyplot as plt
import spacy
import nltk
import re
from nltk import word_tokenize
from nltk.util import ngrams
from sklearn.cluster import KMeans
from collections import Counter
from spacy.lang.pt import Portuguese
nltk.download('punkt')
import string
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.cm as cm
from wordcloud import WordCloud
nlp_pt = spacy.load('pt_core_news_sm')

class clustering:

    def __init__(self,codCVM,arquivoCategoria,especie,dataReferencia,DataEntrega, status, version, modalidade, namePdf ):
        self.codCVM = codCVM
        self.arquivoCategoria = arquivoCategoria
        self.especie = especie
        self.dataReferencia = dataReferencia
        self.DataEntrega = DataEntrega
        self.status = status
        self.version = version
        self.modalidade = modalidade
        self.namePdf = namePdf

    def get_cvm_table(path):
        dt = pd.read_parquet(path)

        return dt

    def extracting_releases_text(df, path):
        for i in df.index:
            file = path +'/'+ df['name_pdf'][i].replace('/','-')
            # print(file)
            pdf = PdfReader(f'{file}')
            text = ""
            for page in pdf.pages:
                text += page.extract_text()
                #extracting text from every page

            df['release_text'][i] = text

    def clear_text(text):
        # remove \n e \r
        c = re.sub(r'\n', ' ', text)
        c = re.sub(r'\r', ' ', c)

        # remove caracteres alfanumericos
        c = c.replace(':', '')
        c = c.replace('/', '')
        c = re.sub(r'R\$', '', c)
        c = re.sub(r'\W', ' ', c)

        # remove espaços adicionais
        c = re.sub(r'\s+', ' ', c)

        # remove espacos adicionais no inicio das frases
        c = re.sub(r'^\s+', '', c)

        # remove espacos adicionais no final das frases
        c = re.sub(r'\s+$', '', c)

        # remove numbers and any other unexpect char
        c= re.sub(r"(?<!\w)([^\w\s]|\d)+(?!\w)", " ", c)
        trans_table = str.maketrans('','', string.digits)
        c = c.translate(trans_table)

        return c.lower() 
    def downloading_stopwords_updating():

        STOP_WORDS = spacy.lang.pt.stop_words.STOP_WORDS #downloading stopwords
        stop_update = ('ao','milhão','bilhão','de','em','e','a','ao','da','do','tudo','bilhão','milhões','bilhões', '4t14','pró','t','h')
        # stop_update = stop_update + update
        STOP_WORDS.add(stop_update)
        return STOP_WORDS
                              
    def remove_stopwords(texto, STOP_WORDS):
        doc = nlp_pt(texto)
        return " ".join(x.text for x in doc if x.text not in STOP_WORDS)
    
    def lematization(texto, STOP_WORDS):
        
        doc = nlp_pt(texto)
        return " ".join(x.lemma_ for x in doc if x.text not in STOP_WORDS)
    
    def exibe_tokens(text):
        col_names = ['ALPHA', 'PUNCT', 'LIKE_NUM', 'POS']
        formatted_text = '{:>16}' * (len(col_names) + 1)
        print('\n', formatted_text.format('INPUT WORD', *col_names), '\n','='*130)

        for token in text:
            output = [token.text, token.is_alpha, token.is_punct, token.like_num, token.pos_]
            print(formatted_text.format(*output))

    def tokenization(text):
        token_list = word_tokenize(text)
        for i in range(len(token_list)):
            try:
                if len(token_list[i]) == 1:
                    # print(roken_list[i])
                    token_list.pop(i)
            except:
                pass
        return  token_list
    
    def tfidf_model(x_labels, STOP_WORDS): 
        tfidf = TfidfVectorizer(
            min_df = 5,
            max_df = 0.95,
            max_features = 8000,
            stop_words = list(STOP_WORDS)
        )
        tfidf.fit(x_labels)
        text = tfidf.transform(x_labels)
        print(f"number of samples: {text.shape[0]}, number of features: {text.shape[1]}")
        print(f"Dispersão {text.nnz / np.prod(text.shape):.3f}") 
        return text, tfidf
    
    def find_optimal_clusters(data, max_k):
        iters = range(2, max_k+1, 2)
        sse = []
        for k in iters:
            if k is not None:
                sse.append(MiniBatchKMeans(n_clusters=k, init_size=1024, batch_size=2048, random_state=42).fit(data).inertia_)
                print('Fit {} clusters'.format(k))
            
            
        f, ax = plt.subplots(1, 1)
        ax.plot(iters, sse, marker='o')
        ax.set_xlabel('Cluster Centers')
        ax.set_xticks(iters)
        ax.set_xticklabels(iters)
        ax.set_ylabel('SSE')
        ax.set_title('SSE by Cluster Center Plot')

        plt.savefig('SSE cluster center plot.png')

    def cluster(text, n_clusters):
        clusters = MiniBatchKMeans(n_clusters=n_clusters, init_size=1024,
            batch_size=2048, random_state=20).fit_predict(text)
        return clusters

    def plot_tsen_pca(data, labels):
        max_label = max(labels)
        try:
            max_items = np.random.choice(range(data.shape[0]), size=3000, replace=False)
        except ValueError:
            max_items = np.random.choice(range(data.shape[0]), size=3000, replace=True)


        # data = np.asarray(data)
        pca = PCA(n_components=2).fit_transform(data[max_items,:].toarray())
        tsne = TSNE().fit_transform(PCA(n_components=50).fit_transform(data[max_items,:].toarray()))
        
        
        idx = np.random.choice(range(pca.shape[0]), size=300, replace=False)
        label_subset = labels[max_items]
        label_subset = [cm.hsv(i/max_label) for i in label_subset[idx]]
        
        f, ax = plt.subplots(1, 2, figsize=(20, 7))
        
        ax[0].scatter(pca[idx, 0], pca[idx, 1], c=label_subset)
        ax[0].set_title('PCA Cluster Plot')
        
        ax[1].scatter(tsne[idx, 0], tsne[idx, 1], c=label_subset)
        ax[1].set_title('TSNE Cluster Plot')

        plt.savefig('PCA E TSNE SCATTERPLOT.png')

        ## MODELO DE CLUSTERIZACAO
    def Kmeans_model(data,data_title,true_k):
        model = KMeans(n_clusters=true_k, init='k-means++', max_iter=200, n_init=10)

        model.fit(data_title)
        labels=model.labels_

        title_cluster_frame=pd.DataFrame(list(zip(data,labels)),columns=['title','cluster'])
        print(title_cluster_frame.sort_values(by=['cluster']))

        return labels, title_cluster_frame

        ## NUNVEM DE PALAVARAS
    def saving_wordcloud_image (title_cluster_frame, data_text, labels, true_k):
        result={'cluster':labels,'wiki':data_text}
        result=pd.DataFrame(result)
        for k in range(0,true_k):
            s=result[result.cluster==k]
            text=s['wiki'].str.cat(sep=' ')
            text=text.lower()
            text=' '.join([word for word in text.split()])
            wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)
            print('Cluster: {}'.format(k))
            print('Titles')
            titles=title_cluster_frame[title_cluster_frame.cluster==k]['title']
            # print(titles.to_string(index=False))
            plt.figure()
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            plt.show()
            plt.savefig('Cluster{}.png'.format(k))

    def get_top_keywords(data, clusters, labels, n_terms):
        df = pd.DataFrame(data.todense()).groupby(clusters).mean()
        df.iterrows()
        for i,r in df.iterrows():
            print('\nCluster {}'.format(i))
            print(','.join([labels[t] for t in np.argsort(r)[-n_terms:]]))
        return df

if __name__ == "__main__":

    data = clustering.get_cvm_table(r'C:\Users\Thiago Jose\Desktop\CODIGO.PY\Aulas-curso\ITAU_QUANT_DESAFIO\Database\text_releases_cvm.parquet')

    data['release_text'] = data['release_text'].apply(lambda x : clustering.clear_text(x))

    stopwords = clustering.downloading_stopwords_updating()
    data['release_text'] = data['release_text'].apply(lambda x : clustering.remove_stopwords(x, stopwords))
    data['release_text'] = data['release_text'].apply(lambda x : clustering.lematization(x, stopwords))
    # data['text'] = data['text'].apply(lambda x : clustering.tokenization(x))
    


    data_transmoed, tfidf = clustering.tfidf_model(data['release_text'], stopwords)
    clustering.find_optimal_clusters(data_transmoed, 20)

    n_clusters = 16
    labels, title_cluster_frame = clustering.Kmeans_model(data_transmoed,data['release_text'],n_clusters)
    clustering.saving_wordcloud_image(title_cluster_frame,data['release_text'],labels,n_clusters)
    # clusters = clustering.cluster(data_transmoed, n_clusters)
    # clustering.plot_tsen_pca(data_transmoed, clusters)

    df_clusters = clustering.get_top_keywords(data_transmoed, n_clusters, tfidf.get_feature_names_out(), 10)

    print(df_clusters)






    