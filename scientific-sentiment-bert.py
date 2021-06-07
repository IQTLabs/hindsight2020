#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Do you want to run this notebook using the uploaded saved checkpoints, or from scratch?
# Note: it takes several hours (more than Kaggle limits) to run from scratch on all the articles from the biorxiv_medrxiv
# dataset, unless you are able to parallelize across multiple CPUs. GPUs are also required for part of the notebook.
RUNTYPE = "mini" # options are {"mini, "full", "cached"}

# Do you want to include paper abstracts in the analysis, or just the main text of the articles?
ABSTRACT = True

print("Running the notebook with the " + RUNTYPE + " dataset...")


# ## 3.1 Edge labelling: training the BERT model on our sentences
# 
# To train BERT, we needed sentences that were labelled by their human-judged utility; we then simply feed these sentences into a pre-trained BERT model (available for download), and fine-tune BERT on our particular task. We obtained ~1000 training sentences by manually going through example sentences from the PolySearch database for diseases we thought were similar to covid19, such as sars, mers, influenza, hiv, malaria, and others, and manually labelling them on a 1-4 scale ('negative', 'neutral', 'weak positive', and 'strong positive'). Some examples are below:
# * Strong positive: The results showed that VOID significantly induced VOID expression in a time- and dose-dependent manner in the VOID. 
# * Weak positive: The results showed that VOID could up-regulate VOID expression time- and dose- dependently in VOID.
# * Neutral: VOID has been persistent in the VOID VOID since 2012. 
# * Negative: The efficacy of VOID fortification against VOID is uncertain in VOID-endemic settings. 
# 
# For binary classification, we converted these scores into positive or not positive.
# 
# We used 900 out of the 1000 samples to train the model, reserving the rest for evaluation. On our evaluation set, our trained model had an F1-score of 94% and an AUC of 93% -- not bad at all given how little time we had! Not terribly surprsing either, given how powerful BERT is compared to other word-embeddings, and that we feel that the task of deciding efficacy valence isn't a particulary difficult one from an NLP perspective. Once we trained BERT on our labelled dataset, we could then label all the sentences we mined thusfar on this Covid19 dataset.
# 
# ### Training runtime
# 
# Training the BERT model below on a GPU takes less than 3 minutes. It also takes some time (around ten minutes) for python to install the libraries and resolve dependencies below.

# In[9]:


# Copyright 2019 Google Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the ssmallpecific language governing permissions and
# limitations under the License.

# Import the correct versions of tensorflow and cuda drivers to match with this code, and install bert.

if RUNTYPE != 'cached':
    #get_ipython().system('apt install -y cuda-toolkit-10-0')
    #get_ipython().system('pip install bert-tensorflow==1.0.1')
    #get_ipython().system('pip install tensorflow==2.0.0')
    #get_ipython().system('pip install tensorflow-gpu==2.0.0')
    from sklearn.model_selection import train_test_split
    import pandas as pd
    import tensorflow as tf
    import tensorflow_hub as hub
    from datetime import datetime
    import numpy as np
    from tqdm import tqdm
    tqdm.pandas()

    #get_ipython().system('pip install bert-tensorflow')
    import bert
    from bert import run_classifier
    from bert import optimization
    from bert import tokenization

    from tensorflow import keras
    import os
    import re

    # check to make sure that this notebook recognizes that we have a GPU
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())
    device_name = tf.test.gpu_device_name()
    if "GPU" not in device_name:
        print("GPU device not found")
    print('Found GPU at: {}'.format(device_name))
else:
    print("Running on cached version without GPU support; using cached version of BERT labels.")

# The code below trains BERT; you can just think of BERT as a module here for doing sentence classification. There are many online resources available for understanding BERT which are better than what I could try to fit in here.

# In[10]:


#####################################################################################################################
#
# TRAIN the BERT model to predict efficacy valence of sentences
#
#####################################################################################################################

# Set the output directory for saving model file
# Optionally, set a GCP bucket location
root_dir = '.'
if RUNTYPE != 'cached':

    OUTPUT_DIR = '.' #@param {type:"string"}

    DATA_COLUMN = 'sentence'
    LABEL_COLUMN = 'polarity'
    label_list = [0,1] # we are doing binary classification

    '''# Code to train and evaluate the model on the training dataset (90/10 split)
    train = temp_df
    test = pd.read_csv(root_dir + "bio-sentiments/all_sentiments_bert.csv", sep=",")
    splitter = np.random.rand(len(test)) < 0.1
    test = test[splitter]
    test_InputExamples = test.apply(lambda x: bert.run_classifier.InputExample(guid=None, 
                                                                       text_a = x[DATA_COLUMN], 
                                                                       text_b = None, 
                                                                       label = x[LABEL_COLUMN]), axis = 1)
    test_features = bert.run_classifier.convert_examples_to_features(test_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)
    '''

    # Our kaggle-competition specific training dataset used to fine-tune our model (made public):
    train = pd.read_csv(root_dir + "/all_sentiments.csv")
    train = train.drop(columns=['Unnamed: 0'])
    train['polarity'] = train['polarity'].apply(lambda x: int(x))

    # Use the InputExample class from BERT's run_classifier code to create examples from the data
    train_InputExamples = train.apply(lambda x: bert.run_classifier.InputExample(guid=None, # Globally unique ID for bookkeeping, unused in this example
                                                                       text_a = x[DATA_COLUMN], 
                                                                       text_b = None, 
                                                                       label = x[LABEL_COLUMN]), axis = 1)

    # This is a path to an uncased (all lowercase) version of BERT
    BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"

    # BERT has its own way to break up a sentence into word-embeddings, using its tokenizer
    def create_tokenizer_from_hub_module():
      with tf.Graph().as_default():
        bert_module = hub.Module(BERT_MODEL_HUB)
        tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
        with tf.Session() as sess:
          vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                                tokenization_info["do_lower_case"]])
      return bert.tokenization.FullTokenizer(
          vocab_file=vocab_file, do_lower_case=do_lower_case)

    tokenizer = create_tokenizer_from_hub_module()

    # We'll set sequences to be at most 128 tokens long.
    MAX_SEQ_LENGTH = 128

    # Convert our training data to InputFeatures that BERT understands.
    train_features = bert.run_classifier.convert_examples_to_features(train_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)

    # Set up the BERT model
    def create_model(is_predicting, input_ids, input_mask, segment_ids, labels,
                     num_labels):
      bert_module = hub.Module(
          BERT_MODEL_HUB,
          trainable=True)
      bert_inputs = dict(
          input_ids=input_ids,
          input_mask=input_mask,
          segment_ids=segment_ids)
      bert_outputs = bert_module(
          inputs=bert_inputs,
          signature="tokens",
          as_dict=True)

      # Use "pooled_output" for classification tasks on an entire sentence.
      # Use "sequence_outputs" for token-level output.
      output_layer = bert_outputs["pooled_output"]
      hidden_size = output_layer.shape[-1].value

      # Create our own layer to tune for politeness data.
      output_weights = tf.get_variable(
          "output_weights", [num_labels, hidden_size],
          initializer=tf.truncated_normal_initializer(stddev=0.02))
      output_bias = tf.get_variable(
          "output_bias", [num_labels], initializer=tf.zeros_initializer())

      with tf.variable_scope("loss"):
        # Dropout helps prevent overfitting
        output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        # Convert labels into one-hot encoding
        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

        predicted_labels = tf.squeeze(tf.argmax(log_probs, axis=-1, output_type=tf.int32))
        # If we're predicting, we want predicted labels and the probabiltiies.
        if is_predicting:
          return (predicted_labels, log_probs)

        # If we're train/eval, compute loss between predicted and actual label
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)
        return (loss, predicted_labels, log_probs)

    # model_fn_builder actually creates our model function
    # using the passed parameters for num_labels, learning_rate, etc.
    def model_fn_builder(num_labels, learning_rate, num_train_steps,
                         num_warmup_steps):
      """Returns `model_fn` closure for TPUEstimator."""
      def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        is_predicting = (mode == tf.estimator.ModeKeys.PREDICT)

        # TRAIN and EVAL
        if not is_predicting:
          (loss, predicted_labels, log_probs) = create_model(
            is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)
          train_op = bert.optimization.create_optimizer(
              loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)
          # Calculate evaluation metrics. 
          def metric_fn(label_ids, predicted_labels):
            accuracy = tf.metrics.accuracy(label_ids, predicted_labels)
            f1_score = tf.contrib.metrics.f1_score(
                label_ids,
                predicted_labels)
            auc = tf.metrics.auc(
                label_ids,
                predicted_labels)
            recall = tf.metrics.recall(
                label_ids,
                predicted_labels)
            precision = tf.metrics.precision(
                label_ids,
                predicted_labels) 
            true_pos = tf.metrics.true_positives(
                label_ids,
                predicted_labels)
            true_neg = tf.metrics.true_negatives(
                label_ids,
                predicted_labels)   
            false_pos = tf.metrics.false_positives(
                label_ids,
                predicted_labels)  
            false_neg = tf.metrics.false_negatives(
                label_ids,
                predicted_labels)
            return {
                "eval_accuracy": accuracy,
                "f1_score": f1_score,
                "auc": auc,
                "precision": precision,
                "recall": recall,
                "true_positives": true_pos,
                "true_negatives": true_neg,
                "false_positives": false_pos,
                "false_negatives": false_neg
            }
          eval_metrics = metric_fn(label_ids, predicted_labels)

          if mode == tf.estimator.ModeKeys.TRAIN:
            return tf.estimator.EstimatorSpec(mode=mode,
              loss=loss,
              train_op=train_op)
          else:
              return tf.estimator.EstimatorSpec(mode=mode,
                loss=loss,
                eval_metric_ops=eval_metrics)
        else:
          (predicted_labels, log_probs) = create_model(
            is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)
          predictions = {
              'probabilities': log_probs,
              'labels': predicted_labels
          }
          return tf.estimator.EstimatorSpec(mode, predictions=predictions)

      # Return the actual model function in the closure
      return model_fn

    # Compute train and warmup steps from batch size
    # These hyperparameters are copied from this colab notebook (https://colab.sandbox.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb)
    BATCH_SIZE = 8
    LEARNING_RATE = 2e-5
    NUM_TRAIN_EPOCHS = 3.0
    WARMUP_PROPORTION = 0.1
    SAVE_CHECKPOINTS_STEPS = 500
    SAVE_SUMMARY_STEPS = 100

    # Compute # train and warmup steps from batch size
    num_train_steps = int(len(train_features) / BATCH_SIZE * NUM_TRAIN_EPOCHS)
    num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)
    # Specify outpit directory and number of checkpoint steps to save
    run_config = tf.estimator.RunConfig(
        model_dir=OUTPUT_DIR,
        save_summary_steps=SAVE_SUMMARY_STEPS,
        save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS)
    model_fn = model_fn_builder(
      num_labels=len(label_list),
      learning_rate=LEARNING_RATE,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps)

    estimator = tf.estimator.Estimator(
      model_fn=model_fn,
      config=run_config,
      params={"batch_size": BATCH_SIZE})
    # Create an input function for training. drop_remainder = True for using TPUs.
    train_input_fn = bert.run_classifier.input_fn_builder(
        features=train_features,
        seq_length=MAX_SEQ_LENGTH,
        is_training=True,
        drop_remainder=True)
    print(f'Beginning Training!')
    current_time = datetime.now()
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
    print("Training took time ", datetime.now() - current_time)
else:
    print("skipped BERT training because cached version running")


quit()


# ## 3.2 Edge labelling: Running the trained BERT model on all the sentences in our edges to get efficacy valence labels
# 
# Once we have trained BERT, we can then use it to label our sentences. First, we break up the sentence pairs into individual sentences, as this is what BERT was originally trained on. Then, we generate labels for each of the sentences, and store these in our dataframe of edges.
# 
# ### Runtime
# BERT should be run with a GPU, and can take about an hour on the full dataset, and runs in a few minutes on the mini dataset. If you are running this notebook with the cached setting on, it will skip running BERT on the GPU, and will just load the pre-labelled sentences that we created with the same code offline.

# In[ ]:


#####################################################################################################################
#
# Run the trained BERT model to predict efficacy valence of sentences from our edges
#
#####################################################################################################################

if RUNTYPE != 'cached':

    # Our test data contains two senteces for each edge; we need to split these up using the separator below, and then record
    # a score for each of the two sentences in the dataframe
    separator = '------'
    def splitter(sentence, separator, index):
       if separator in sentence:
           return sentence.split(separator)[index]
       else:
           return sentence

    def getPrediction(in_sentences):
      labels = ["Negative", "Positive"]
      input_examples = [run_classifier.InputExample(guid="", text_a = x, text_b = None, label = 0) for x in in_sentences] # here, "" is just a dummy label
      input_features = run_classifier.convert_examples_to_features(input_examples, label_list, MAX_SEQ_LENGTH, tokenizer)
      predict_input_fn = run_classifier.input_fn_builder(features=input_features, seq_length=MAX_SEQ_LENGTH, is_training=False, drop_remainder=False)
      predictions = estimator.predict(predict_input_fn)
      return [(sentence, prediction['probabilities'], labels[prediction['labels']]) for sentence, prediction in zip(in_sentences, predictions)]

    def util_calc(x):
        if x == 'Negative': 
            return 0
        return 1

    # to avoid running BERT forever, find the unique senteces from all sentence pairs, and only label those
    test = pd.read_csv("voided_edges.csv") 
    test['polarity'] = test['utility']
    sentenceHash = {}
    sentences = test['sentence']

    sep="------"
    single_sentences = []
    for a in sentences:
        splitt = a.split(sep)
        if len(splitt) == 2: #there are a few outliers that had three, just ignore them
            a, b = splitt
            single_sentences.append(a)
            single_sentences.append(b)
    single_sentences = list(set(single_sentences))

    # prepare data for BERT of just single, unique sentences
    labels = [0 for s in single_sentences]
    single_test = pd.DataFrame(list(zip(single_sentences, labels)), columns=['sentence','polarity'])
    test_InputExamples = single_test.apply(lambda x: bert.run_classifier.InputExample(guid=None, 
                                                                           text_a = x[DATA_COLUMN], 
                                                                           text_b = None, 
                                                                           label = x[LABEL_COLUMN]), axis = 1)
    test_features = bert.run_classifier.convert_examples_to_features(test_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)
    print("finished tokenizing unique sentences...")

    # pass the tokenized sentences to BERT, label them
    test_input_fn = run_classifier.input_fn_builder(
            features=test_features,
            seq_length=MAX_SEQ_LENGTH,
            is_training=False,
            drop_remainder=True)
    preds = getPrediction(single_test['sentence'])
    preds = [util_calc(p[2]) for p in preds]

    def hash_helper(x):
        if x in sentenceHash.keys(): return sentenceHash[x]
        else:
            print(x)
            return 'ERROR'

    test.to_csv('worksaver.csv')

    # find the single sentences in the original sentence pairs, and assign them their two labels
    for i, s in enumerate(single_sentences):
        sentenceHash[s] = preds[i]
    test = test[test['sentence'].map(lambda x: len(x.split(sep)) != 3)]
    test['sentence0'] = test['sentence'].apply(lambda x: x.split(sep)[0])
    test['sentence1'] = test['sentence'].apply(lambda x: x.split(sep)[1])
    test['utility'] = test['sentence0'].apply(lambda x: hash_helper(x))
    test['utility2'] = test['sentence1'].apply(lambda x: hash_helper(x))
    test = test[test['utility'] != 'ERROR']
    test = test[test['utility2'] != 'ERROR']

    print(test.head())
    print(len(test))
    print((test.utility.value_counts()))
    print((test.utility2.value_counts()))
    test.to_csv("labled_one_million.csv")
    print("finished labelling all sentences with BERT.")
else:
    test = pandas.read_csv(root_dir + "labelled-cached/labelled_edges_cached.csv")
    test = test[test['utility'] != 'ERROR']
    test = test[test['utility2'] != 'ERROR']
    print("skipped running BERT on sentences because we're running the cached version.")


# <a id="section4"></a>
# # 4. Drawing the graph
# 
# Once we have all our edges labelled with the efficacy valence of the sentences that generated them, we can then build our knowledge graph using these scores as weights.
# 
# ## 4.1 Drawing the graph: Cleaning the raw edges
# 
# We first do some housekeeping around the edges dataframe, converting strings to numerics as needed, coalescing the disease synonyms into a single node (we only did this for covid19, but had we more time we should have done it for all drugs/keywords), and manually provide labels for the edges that came directly from PolySearch results, as we didn't create them using any source sentences (we could have in theory, but just took this shortcut given our time constraints).
# 
# If you have many edges (over a million), this can take around ten minutes.

# In[ ]:


import numpy as np
import pandas as pd
import networkx as nx
import ast
import matplotlib.pyplot as plt
import datetime
from random import randint
import copy
import os
#from tqdm import tqdm
#tqdm.pandas()
import sys
sys.executable

edges = test

diseaseSynonyms = ["Covid19", "Covid-19", "coronavirus", "SARS-coronavirus", "2019-nCoV", "SARS-CoV-2", 'COVID19', "Coronavirus", 'covid19', 'sars-cov-2', '2019-ncov',
	"sars-coronavirus", '2019ncov', 'ncov2019', 'ncov-2019', 'covid-19']
edges['node1'] = edges['node1'].apply(lambda x: "covid19" if x in diseaseSynonyms else x)
edges['node2'] = edges['node2'].apply(lambda x: "covid19" if x in diseaseSynonyms else x)

def isDrug(n1, n2, paperUID):
    return drugLookup(n1) or drugLookup(n2) or paperUID == 'PolySearch'

def cleanPoly(utility, paper):
    if paper == 'PolySearch':
        return 1
    return int(utility)

# convert the columns into numeric types as needed
edges['isDrug'] = edges[['node1', 'node2', 'paperUID']].apply(lambda x: isDrug(*x), axis=1)


edges['utility'] = edges[['utility', 'paperUID']].apply(lambda x: cleanPoly(*x), axis=1)
edges['utility2'] = edges[['utility2', 'paperUID']].apply(lambda x: cleanPoly(*x), axis=1)
edges['paperCitationCount'] = edges['paperCitationCount'].apply(lambda x: 0 if x == '-1' else int(x))
edges['context_sum'] = edges.context.apply(lambda x: sum(eval(x)))

# Below, we make some (admittedly minimal and ad-hoc) attempts to coalesce synonyms into single nodes.
edges['node1'] = edges['node1'].apply(lambda x: "receptor" if 'receptor' in x else x)
edges['node2'] = edges['node2'].apply(lambda x: "receptor" if 'receptor' in x else x)
edges['node1'] = edges['node1'].apply(lambda x: "lopinavir" if 'lopinavir' in x else x)
edges['node2'] = edges['node2'].apply(lambda x: "lopinavir" if 'lopinavir' in x else x)
edges['node1'] = edges['node1'].apply(lambda x: "hiv" if 'hiv' in x else x)
edges['node2'] = edges['node2'].apply(lambda x: "hiv" if 'hiv' in x else x)
edges['node1'] = edges['node1'].apply(lambda x: "cytokine" if 'cytokine' in x else x)
edges['node2'] = edges['node2'].apply(lambda x: "cytokine" if 'cytokine' in x else x)
edges['node1'] = edges['node1'].apply(lambda x: "china" if 'chinese' in x else x)
edges['node2'] = edges['node2'].apply(lambda x: "china" if 'chinese' in x else x)
edges['node1'] = edges['node1'].apply(lambda x: "trial" if 'trial' in x else x)
edges['node2'] = edges['node2'].apply(lambda x: "trial" if 'trial' in x else x)
edges['node1'] = edges['node1'].apply(lambda x: "interferon" if 'interferon' in x else x)
edges['node2'] = edges['node2'].apply(lambda x: "interferon" if 'interferon' in x else x)

edges.head()


# ## 4.2 Drawing the graph: creating fly-out edges for final nodes
# 
# We want the user to be able to drill down into each node in our graph, beyond just the most meaningful drugs and topics we find. Therefore, we will get additional edges where node1 matches the final node in our graph, and node2 is a drug and/or topic that occurs above some threshold in our entire dataset. 

# In[ ]:


# calculate the frequency for all node2 keywords
index = list(edges['node2'].value_counts().index)
counts = list(edges['node2'].value_counts())

ctr = 0
while ctr < len(index):
    print(str(counts[ctr]) + "\t" + index[ctr] + " "  )
    ctr += 1    
value_counts = {index[i]: counts[i] for i in range(len(index))}

# mine additional edges from a source node1, where the node2 values are a drug, and occur above some threshold
threshold_additional = 10
def getAdditionalEdges(node):
    mini1 = edges[edges['node2'] == node]
    mini1 = mini1[mini1['utility'] == 1]
    mini2 = edges[edges['node1'] == node]
    mini2 = mini2[mini2['utility'] == 1]
    result = set(list(mini1['node1']) + list(mini2['node2']))
    cleaned = []
    for r in result:
        if value_counts[r] > threshold_additional and drugLookup(r):
            cleaned.append(r)
    #print((cleaned))
    return cleaned

print(edges.isDrug.value_counts())
print(len(edges))


# ## 4.3 Drawing the graph: Generate the goal graph as a baseline
# 
# Recall, we used a reputable literature survey paper on Covid19 to generate a goal graph that we can compare our graph to. Below, we encode the drugs and concepts from that survey paper into a graph, and provide a function that can measure how many nodes two graphs have in common.
# 
# We commented out any nodes (drugs) below that never showed up in the dataset, as we shouldn't use them to measure the quality of our graphs (since it would be impossible for us to ever have seen them).

# In[ ]:


G_goal = nx.Graph()
G_goal.add_edge('remdesivir', 'covid19')
G_goal.add_edge('remdesivir', 'trials')
G_goal.add_edge('remdesivir', 'sars')
G_goal.add_edge('remdesivir', 'mers')
G_goal.add_edge('remdesivir', 'nausea')
G_goal.add_edge('remdesivir', 'vomiting')
G_goal.add_edge('remdesivir', 'transaminase')
G_goal.add_edge('remdesivir', 'renal')
G_goal.add_edge('chloroquine', 'hydroxychloroquine')
G_goal.add_edge('chloroquine', 'sars')
G_goal.add_edge('hydroxychloroquine', 'sars')
G_goal.add_edge('hydroxychloroquine', 'emergency')
G_goal.add_edge('chloroquine', 'emergency')
G_goal.add_edge('chloroquine', 'toxicity')
G_goal.add_edge('hydroxychloroquine', 'toxicity')
G_goal.add_edge('chloroquine', 'interactions')
G_goal.add_edge('hydroxychloroquine', 'interactions')
G_goal.add_edge('chloroquine', 'dosing')
G_goal.add_edge('hydroxychloroquine', 'dosing')
G_goal.add_edge('hydroxychloroquine', 'trials')
G_goal.add_edge('hydroxychloroquine', 'fever')
G_goal.add_edge('hydroxychloroquine', 'cough')
G_goal.add_edge('hydroxychloroquine', 'chest')
G_goal.add_edge('chloroquine', 'symptoms')
G_goal.add_edge('chloroquine', 'china')
G_goal.add_edge('hydroxychloroquine', 'azithromycin')
G_goal.add_edge('azithromycin', 'rna')
G_goal.add_edge('azithromycin', 'trials')
G_goal.add_edge('interleukin-6', 'covid19')
G_goal.add_edge('interleukin-6', 'anecdotal')
G_goal.add_edge('interleukin-6', 'covid19')
G_goal.add_edge('interleukin-6', 'tocilizumab')
#G_goal.add_edge('interleukin-6', 'sarilumab')
#G_goal.add_edge('interleukin-6', 'siltuximab')
G_goal.add_edge('tocilizumab', 'trials')
#G_goal.add_edge('sarilumab', 'trials')
#G_goal.add_edge('siltuximab', 'trials')
G_goal.add_edge('plasma', 'emergency')
G_goal.add_edge('plasma', 'covid19')
G_goal.add_edge('plasma', 'emergency')
G_goal.add_edge('plasma', 'oxygenation')
G_goal.add_edge('favipiravir', 'influenza')
G_goal.add_edge('favipiravir', 'trials')
G_goal.add_edge('favipiravir', 'covid19')
G_goal.add_edge('favipiravir', 'clearance')
G_goal.add_edge('favipiravir', 'trials')
G_goal.add_edge('lopinavir-ritonavir', 'sars')
G_goal.add_edge('lopinavir-ritonavir', 'hiv')
G_goal.add_edge('lopinavir-ritonavir', 'mers')
G_goal.add_edge('lopinavir-ritonavir', 'trials')

# measures how similar two graphs are based on shared nodes.
def compareGraphs(G_baseline, G_test):
    totalNodes = 0
    foundNodes = 0
    for n in G_baseline.nodes:
        if n in G_test.nodes:
            foundNodes += 1
        totalNodes += 1
    return foundNodes * 100.0 / totalNodes

f = plt.figure(figsize=(15,15))
nx.draw(G_goal,arrows=None, with_labels=True)


# ## 4.4 Drawing the graph: Pruning edges by weight
# 
# Our algorithm goes through all the edges, and calculates an aggregate weight for each unique node1-node2 pair it sees (order doesn't matter, we sort the node names alphabetically). Recall, the same edge may have been generated more than once if it appears multiple times in the articles.
# 
# We will build the graph using a weight of our choice:
# * "polarity": will take the average of all the efficacy valence scores across the same edge, using the max score of the sentence-pair
# * "polarity_sum" : sums, rather than average, all efficacy valence scores across the same edge, using the max score of the sentence-pair
# * "interestingWords" : sums up all the interesting words ever seen for this edge, including duplicates
# * "citations" : calculates the average citation count for this edge's sentences/papers, NOT weighted by publication year (you could change this)
# * "drug" : sums all the times at least one keyword in the edge was a drug
# * otherwise, it just sums the all the above features (in an unintelligent way; you could optimize this, we ran out of time and were satisfied with our results with "polarity_sum")
# 
# The code below implements these functions.

# In[ ]:


# calculates the weight the first time an edge is seen, using the choice specified by the user
def calcInitialWeight(interestingWords, paperCitationCount, paperYear, polarity, polarity2, drug, choice):
    if choice == 'interestingWords':
        return eval(interestingWords)
    elif choice == 'citations':
        return [int(paperCitationCount)]
    elif choice == 'polarity':
        return [max(int(polarity), int(polarity2))]
    elif choice == 'polarity_sum':
        return max(int(polarity), int(polarity2))
    elif choice == 'drug':
        if drug == True:
            return 1
        else:
            return 0
    else:
        drug = 1 if drug == True else 0
        return sum(eval(interestingWords)) + int(polarity) + int(paperCitationCount) + drug

# updates the respective edge weight
def updateWeight(weight, interestingWords, paperCitationCount, paperYear, polarity, polarity2, drug, choice):
    if choice == 'interestingWords':
        ctr = 0
        interestingWords = eval(interestingWords)
        while ctr < len(interestingWords):
            if interestingWords[ctr] == 1:
                weight[ctr] = 1
            ctr += 1
        return weight
    elif choice == 'citations':
        weight.append(int(paperCitationCount))
        return weight
    elif choice == 'polarity':
        weight.append(max(int(polarity), int(polarity2)))
        return weight
    elif choice == 'polarity_sum':
        weight += (max(int(polarity), int(polarity2)))
        return weight    
    elif choice == 'drug':
        if drug == True:
            return 1 + weight
        else:
            return weight
    else:
        drug = 1 if drug == True else 0
        return weight + sum(eval(interestingWords)) + int(polarity) + int(paperCitationCount) + drug

def sorted(x, y):
    if x < y:
        return x, y
    return y, x
    
#function to add edges, summing one-hot encoding of context using numpy arrays 
def edge_add(node1, node2, paperYear, context, paperCitationCount, sentiment, polarity, drug, polarity2, context_sum, G, choice):
    a, b = sorted(node1,node2)
    if type(paperYear) != type('string'):
        paperYear = "2020"

    if G.has_edge(a, b):
        old_weight = G[a][b]['weight']
        G[a][b]['weight'] = updateWeight(old_weight, context, paperCitationCount, paperYear, polarity, polarity2, drug, choice)
    else:
        G.add_edge(a, b, weight=calcInitialWeight(context, paperCitationCount, paperYear, polarity, polarity2, drug, choice))


# ## 4.5 Drawing the graph: generate the graph(s)
# 
# We can now generate one or more graphs using the pruned edges. The code below allows you to conduct a GridSearch on the potential weighting schemes and thresholds, finding the graph that has the highest overlap in nodes with the goal graph.
# 
# Currently, the code below is set to use the "polarity_sum" weighting scheme (efficacy valence) with a threshold of 20, but you could try the other weighting schemes and thresholds and find the best graph automatically.
# 
# At this point, we restrict this graph to only be built from edges that have at least one drug as a node, to limit its size, but you could play around with this!

# In[ ]:


# restrict this graph to just covid19 treatments
mini_edges = edges[edges['isDrug'] == True]
print('length of drug edges: ', len(mini_edges))

# removes all edges that do not meet the minimum weight requirements
def cleanGraphUsingThreshold(G, threshold, choice):
    for e in G1.edges:
        weights = G1.edges[e]['weight']
        # if we need to average the scores in a list (for some weight calculations)
        if choice == 'polarity' or choice == 'interestingWords' or choice == 'citations':
            mean = sum(weights) * 1.0 / len(weights)
            G1.edges[e]['weight'] = mean
            
        if G1.edges[e]['weight'] < threshold:
            G.remove_edge(e[0], e[1])

    # remove all nodes with no edges after cleaning
    nodes = list(G.nodes).copy()
    for n in nodes:
        if len(list(G.adj[n])) < 1:
            G.remove_node(n)

# set the weighting scheme and thresholds to build a graph, using the GridSearch below
choices = ['polarity_sum'] # you would normally provide a list here if you want to do GridSearch
if RUNTYPE == 'mini': # if we're just building a graph from two articles, there won't be much there, so keep weights low
    thresholds = [0]
else:
    thresholds = [20] # you would normally provide a list here if you want to do GridSearch
    
# use GridSearch to find the optimum thresholds and weighting scheme, compared to the goal graph you created earlier
for choice in choices:
    for thresh in thresholds:
        subset = mini_edges.copy(deep=True)
        G1 = nx.Graph()
        subset[['node1', 'node2', 'paperYear', 'context', 'paperCitationCount', 'sentiment', 'utility', 'isDrug', 'utility2', 'context_sum']].apply(lambda x: edge_add(*x, G1, choice), axis = 1)
        cleanGraphUsingThreshold(G1, thresh, choice)
        print('score: ' + str(compareGraphs(G_goal, G1))[:3] + " choice: " + choice + " thresh: " + str(thresh))


# Let's take a look at our draft graph, knowing that we will still need to do some pruning:

# In[ ]:


# draws the first graph we made
f = plt.figure(figsize=(15,15))
nx.draw(G1,arrows=None, with_labels=True)


# ## 4.6 Drawing the graph: Cleaning the best graph
# 
# Given the little time we had for this project, we could have done a more formal and/or better job of identifying stopwords that should not be nodes, based on word frequency, parts of speech, etc. Here, we have a function that you can customize to remove any superfluous nodes and edges from the graph that you are confident aren't really meaningful.
# 
# We also remove any nodes that don't have any edges coming out of them after the pruning process.

# In[ ]:


clutter = ['membrane', 'rights', 'multiple', 'therapeutics', 'condition', 'population', 'screening', 'limit', 'growth',
          'intracellular', 'data', 'strains', 'therapeutic', 'time', 'nuclei', 'evaluation', 'work', 'transport', 'conductance',
          'immune', 'inhibition', 'degradation', 'antibodies', 'receptor', 'populations', 'mast', 'family', 'alleles', 'production',
          'synthesis', 'future', 'cost', 'low', 'kinetics', 'tablets', 'alpha', 'regulation', 'unknown', 'fraction', 'nature',
          'pathogenesis', 'selective', 'strain', 'short', 'expansion', 'play', 'observation', 'sensitivity', 'males', 'females',
          'injury', 'factor', 'transcription', 'aged', 'history', 'secretion', 'survival', 'plays', 'antibody', 'culture',
          'behavior', 'formation', 'recombination', 'ubiquitination', 'vertebrates', 'beta', 'injection', 'white', 'up-regulation',
          'heat', 'enzymes', 'memory', 'electron', 'transfer', 'eating', 'ions', 'transfection', 'conjugated', 'vertebrate', 'abl',
          'acid', 'sodium', 'calcium', 'cations', 'ant', 'ice', 'rain', 'air', 'potassium', 'hydrogen', 'acids', 'magnesium',
          'charge', 'zinc']

# remove any nodes manually that should have ideally be in our stoplist in the first place.
G2a = nx.Graph()
for edge in G1.edges:
    if edge[0] not in clutter and edge[1] not in clutter:
        G2a.add_edge(edge[0], edge[1], weight=1)
        
# also remove the nodes we see above that aren't connected to the main graph, for legibility
# depends on what you set your threshold to above
outliers = ['digoxin', 'vancomycin', 'stress', 'vagina', 'ethyl', 'prednisolone', 'kinase']
G2b = nx.Graph()
for edge in G2a.edges:
    if edge[0] not in outliers and edge[1] not in outliers:
        G2b.add_edge(edge[0], edge[1], weight=1)

# remove any nodes that no longer have any edges after pruning
G2 = copy.deepcopy(G2b)
for n in nx.isolates(G2b):
    G2.remove_node(n)
    
f = plt.figure(figsize=(15,15))
nx.draw(G2,arrows=None, with_labels=True)
print(G2.nodes)


# ## 4.7 Drawing the graph: Displaying the graph
# 
# To display our graph, our code will generate an image for each node that contains its fly-out graph; we create a directory to store these images that is separate from the cached directory.
# 
# Recall, the fly-out graphs for each drug show current and additional connections not in the graph above; we chose to use fly-outs on individual nodes to reduce the clutter in the previous graph, and to allow for focused attention on edges for a fly-out node that may be below the global threshold setting chosen earlier.
# 

# In[ ]:


# create/clean the directory (was too lazy to look up how to check if dir is there)
try:
    os.system("mkdir images_live")
except:
    os.system("rm -r images_live")
    

# call the code to generate fly-out nodes from every node in the main graph, all in a single graph (will be very dense!)
G3 = copy.deepcopy(G2)
for node in G2.nodes:
    for n in getAdditionalEdges(node):
        if str(node) not in clutter:
            G3.add_edge(node, n, weight=0)

# generate individual fly-out graphs that mimic what will take place when someone "clicks" an individual node
# most of the effort here is to customize the node and edge coloroing depending on what the selected node is
for seed_node in G2.nodes:
    G_specific = copy.deepcopy(G3)
    node_colors = {}
    labels = {}
    sizes = {}
    for node in G_specific.nodes:
        node_colors[node] = "aliceblue"
        labels[node] = ""
        sizes[node] = value_counts[node] / 10.0 + 100
    edge_colors = {}
    for edge in G_specific.edges:
        if edge[0] == seed_node or edge[1] == seed_node:
            node_colors[edge[0]] = "deepskyblue"
            node_colors[edge[1]] = "deepskyblue"
            labels[edge[0]] = str(edge[0])
            labels[edge[1]] = str(edge[1])
            edge_colors[edge] = "black"
            sizes[edge[0]] = value_counts[edge[0]] / 10.0 + 100
            sizes[edge[1]] = value_counts[edge[1]] / 10.0 + 100
            
    sizes['covid19'] = 400
    node_colors['covid19'] = "red"

    nodelist = []
    node_color = []
    node_size = []
    for k in node_colors.keys():
        nodelist.append(k)
        node_color.append(node_colors[k])
        node_size.append(sizes[k])
    edgelist = []
    edge_color = []
    for k in edge_colors.keys():
        edgelist.append(k)
        edge_color.append(edge_colors[k])
    f = plt.figure(figsize=(15,15))
    nx.draw_spring(G_specific,arrows=None, with_labels=True, ax=f.add_subplot(111), nodelist=nodelist,
                   edgelist=edgelist,node_size=node_size,node_color=node_color,node_shape='o', alpha=1.0,
                   cmap=None, vmin=None,vmax=None, linewidths=None, width=1.0, edge_color=edge_color,
                   edge_cmap=None, edge_vmin=None,edge_vmax=None, style='solid', labels=labels, font_size=12, 
                   font_color='black', font_weight='normal', font_family='sans-serif', label='COVID19 treatments/drugs')
    f.savefig("./images_live/" + str(seed_node) + ".png")


# colorize the core graph we made earlier and save it to the same directory as the other graphs
node_colors = {}
labels = {}
sizes = {}
edge_colors = {}
for edge in G2.edges:
    node_colors[edge[0]] = "deepskyblue"
    node_colors[edge[1]] = "deepskyblue"
    labels[edge[0]] = str(edge[0])
    labels[edge[1]] = str(edge[1])
    edge_colors[edge] = "black"
    sizes[edge[0]] = value_counts[edge[0]] / 10.0 + 100
    sizes[edge[1]] = value_counts[edge[1]] / 10.0 + 100
sizes['covid19'] = 400
node_colors['covid19'] = "red"
nodelist = []
node_color = []
node_size = []
for k in node_colors.keys():
    nodelist.append(k)
    node_color.append(node_colors[k])
    node_size.append(sizes[k])
edgelist = []
edge_color = []
for k in edge_colors.keys():
    edgelist.append(k)
    edge_color.append(edge_colors[k])
f = plt.figure(figsize=(15,15))
nx.draw_spring(G2,arrows=None, with_labels=True, ax=f.add_subplot(111), nodelist=nodelist,
                   edgelist=edgelist,node_size=node_size,node_color=node_color,node_shape='o', alpha=1.0,
                   cmap=None, vmin=None,vmax=None, linewidths=None, width=1.0, edge_color=edge_color,
                   edge_cmap=None, edge_vmin=None,edge_vmax=None, style='solid', labels=labels, font_size=12, 
                   font_color='black', font_weight='normal', font_family='sans-serif', label='COVID19 treatments/drugs')
f.savefig("./images_live/CORE_RESULTS.png")


# ## 4.8 Drawing the graph: Preparing a dataframe to select papers/sentences based on a selected node
# 
# This is meant as a placeholder for future work that can use automated text summarization and highlighting to coalesce the "search results" we show here
# 

# In[ ]:


columns = ['node1', 'node2', 'paperUID', 'paperYear', 'context', 'paperCitationCount', 'sentiment', 
               'sentence', 'utility', 'isDrug', 'level']
raw_edges = edges.copy(deep=True)

raw_edges['utility1'] = edges['utility']
raw_edges['utility2'] = edges['utility2']

def max1(a, b):
    a = int(a)
    b = int(b)
    if a > b:
        return a
    return b

raw_edges['utility'] = raw_edges[['utility1', 'utility2']].apply(lambda x: max1(*x), axis=1)
    
def mineSentences(keyword, edges, utility):
    subset1 = edges[edges['node1'] == keyword]
    subset2 = edges[edges['node2'] == keyword]
    joined = pd.concat([subset1, subset2])
    joined = joined.reset_index()
    joined = joined.drop(columns=['sentence', 'sentence0', 'sentence1', 'utility1', 'utility2', 'index', 'paperYear', 'context', 'paperCitationCount', 'sentiment', 'isDrug'])
    joined = joined[joined['utility'] >= utility]
    joined = joined.drop_duplicates()
    display(joined)
    #display(set(joined['paperUID']))


# ## 4.9 Drawing the graph: Drawing the interactive graph
# 
# We will use some widgets to allow for a drop-down menu to select either the CORE_RESULTS (default), or select one of the nodes in that graph and show its fly-out edges.

# In[ ]:


import ipywidgets as widgets
from ipywidgets import interact, interact_manual
from IPython.display import Image, display, HTML

# choose which directory to load images from <--- I think this should be rewritten to be live, based on code above
if RUNTYPE != 'cached':
    files = os.listdir("images_live/")
    cleaned = []
    for f in files:
        cleaned.append(f.split(".png")[0])
    files = cleaned
    files.pop(files.index('CORE_RESULTS'))
    files = ['CORE_RESULTS'] + files
else:
    files = ['CORE_RESULTS'] + list(G2.nodes)
directory = widgets.Dropdown(options=files, description='node:')
images = widgets.Dropdown(options=files)

def update_images(*args):
    images.options = files

directory.observe(update_images, 'value')

def show_images(file):
    display(Image('./images_live/' + file + '.png'))

    drug = file
    @interact
    def show_articles_more_than(node=drug, utility=1):
        mineSentences(node, raw_edges, utility)
    
_ = interact(show_images, file=directory)

