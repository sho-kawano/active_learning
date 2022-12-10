from sklearn.datasets import fetch_20newsgroups
categories = ['alt.atheism', 'soc.religion.christian', 'rec.sport.baseball', 'comp.graphics', 'sci.med']  

mydata_train = fetch_20newsgroups(subset='train', shuffle=True, categories=categories, 
                                  remove = ('headers', 'footers', 'quotes'), random_state=42)

mydata_test = fetch_20newsgroups(subset='test', shuffle=True, categories=categories, 
                                  remove = ('headers', 'footers', 'quotes'), random_state=42)

print('size of training set: %s' % (len(mydata_train ['data'])))
print('size of validation set: %s' % (len(mydata_test['data'])))
print('classes: %s' % (mydata_train.target_names))

# ------------------------------------------------------------------------------------------
import pandas as pd
mydata_train_df = pd.DataFrame({'data': mydata_train.data, 'target': mydata_train.target})

X = mydata_train.data
y = mydata_train.target

# ------------------------------------------------------------------------------------------
import numpy as np
n_per_cat = 4
np.random.seed(777)
initial_idx = []
for cat in range(len(categories)):
    initial_idx.append(mydata_train_df[mydata_train_df.target==cat][0:n_per_cat].index) 

initial_idx = [item for sublist in initial_idx for item in sublist]
X_train = [X[i] for i in initial_idx]
y_train = [y[i] for i in initial_idx]

# ------------------------------------------------------------------------------------------
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier


rf_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', RandomForestClassifier())])

# ------------------------------------------------------------------------------------------

from modAL.models import ActiveLearner
from sklearn import metrics


def al_sim(active_learner, n_queries):
    # active learning
    query_indices = []
    accuracy = []
    inclass_acc = []
    # ------ Learning Loops ---------  
    for k in range(n_queries):
        query_idx, query_instance = active_learner.query(X)
        active_learner.teach([X[query_idx[0]]], [y[query_idx[0]]])
        # save query indices 
        query_indices.append(query_idx[0]) 
        # save accuracy index
        accuracy.append(active_learner.score(X=X, y=y))
        # precision / recall 
        prediction = active_learner.predict(X=X)
        precision_recall = metrics.precision_recall_fscore_support(y, prediction)
        inclass_acc.append(precision_recall[2])
    
    query_cat = mydata_train_df.iloc[query_indices, ].target
    return np.array(accuracy), np.array(query_cat), np.array(inclass_acc)
    #np.array(precision), np.array(recall)


def al_sims(query_strategy, clf, n_queries, n_sims):
    sim_accuracy = []
    sim_qcat = []
    sim_inclass = []
    for i in np.arange(n_sims):
        active_learner = ActiveLearner(estimator=clf, 
            query_strategy=query_strategy, 
            X_training=X_train, y_training=y_train)
        accuracy, query_cat, inclass = al_sim(active_learner, n_queries)
        #precision, recall
        sim_accuracy.append(accuracy)
        sim_qcat.append(query_cat)
        sim_inclass.append(inclass)
    accuracy_mat = np.concatenate(sim_accuracy, axis=0) 
    accuracy_mat = accuracy_mat.reshape(n_sims, n_queries) 
    qcat_mat = np.concatenate(sim_qcat, axis=0) 
    qcat_mat = qcat_mat.reshape(n_sims, n_queries) 
    #                              note: sim x cat  x queries 
    return(accuracy_mat, qcat_mat, np.array(sim_inclass)) 


def run_sim(sampling_strategy,  file_name, n_queries, n_sims,):
    results = al_sims(sampling_strategy, rf_clf, n_queries, n_sims)
    np.save(file_name + '.npy', results[0:2])   
    np.save(file_name + '_accuracy.npy', results[2])
    print("simulation done!")
    print(file_name + " files created.")


from modAL.uncertainty import entropy_sampling, margin_sampling
def random_sampling(classifier, X_pool):
    n_samples = len(X_pool)
    query_idx = np.random.choice(range(n_samples))
    return [query_idx], X_pool[query_idx]

# ------------------------------------------------------------------------------------------
from multiprocessing import Process, set_start_method
set_start_method("fork")

n_queries = 40
n_sims = 60

sim1 = Process(target=run_sim, args=(margin_sampling, 'margin_results5', n_queries, n_sims))
sim2 = Process(target=run_sim, args=(random_sampling, 'random_results5', n_queries, n_sims))
sim3 = Process(target=run_sim, args=(entropy_sampling, 'entropy_results5', n_queries, n_sims))

sim1.start()
sim2.start()
sim3.start()

sim1.join()
sim2.join()
sim3.join()