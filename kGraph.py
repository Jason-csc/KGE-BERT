import spacy
from typing import List
from datetime import datetime
import numpy as np
import torch

import nltk

from nltk.tokenize import word_tokenize

import sys
PATH = "/content/drive/MyDrive/EECS487/project/deepwalk/deepwalk"
sys.path.append(PATH)

from process import process



nlp = spacy.load('en_core_web_sm')

import gensim
import gensim.downloader
embed = gensim.downloader.load('word2vec-google-news-300')
num = 0
average_vec = np.zeros(300)
for w in embed.vocab:
    average_vec += embed[w]
    num += 1
average_vec /= num


def get_entities(sent):
    ## chunk 1
    ent1 = ""
    ent2 = ""

    prv_tok_dep = ""
    prv_tok_text = ""

    prefix = ""
    modifier = ""

    for tok in nlp(sent):
        if tok.dep_ != "punct":
            if tok.dep_ == "compound":
                prefix = tok.text
                if prv_tok_dep == "compound":
                    prefix = prv_tok_text + " " + tok.text

            if tok.dep_.endswith("mod") == True:
                modifier = tok.text
                if prv_tok_dep == "compound":
                    modifier = prv_tok_text + " " + tok.text

            if tok.dep_.find("subj") == True:
                ent1 = modifier + " " + prefix + " " + tok.text
                prefix = ""
                modifier = ""
                prv_tok_dep = ""
                prv_tok_text = ""

            if tok.dep_.find("obj") == True:
                ent2 = modifier + " " + prefix + " " + tok.text

            prv_tok_dep = tok.dep_
            prv_tok_text = tok.text

    return [ent1.strip(), ent2.strip()]



PATH = "/content/drive/My Drive/EECS487/project/deepwalk/deepwalk"

def buildGraph(context:str):
    id_to_node = {}
    node_to_id = {}
    nodeID = 1
    doc = nlp(context)
    lines = []
    for stc in doc.sents:
        sentence = stc.text
        node1, node2 = get_entities(sentence)
        if len(node1) == 0 or len(node2) == 0:
            continue
        if not node1 in node_to_id:
            node_to_id[node1] = nodeID
            id_to_node[nodeID] = node1
            nodeID += 1
        if not node2 in node_to_id:
            node_to_id[node2] = nodeID
            id_to_node[nodeID] = node2
            nodeID += 1
        id1 = node_to_id[node1]
        id2 = node_to_id[node2]
        lines.append(f"{id1} {id2}\n")
    if len(lines) == 0:
        return None, None, None, None
    date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
    filename = f"{PATH}/tmp/kg_{date}"
    with open(filename,'w') as f:
        f.writelines(lines)
    model, num_node = process(filename)
    return model, num_node, id_to_node, node_to_id
    


def getEmbedding(sentence):
    '''Average the word embedding in sentence to get sentence embedding'''
    vec = np.zeros(300)
    n = 0
    for token in word_tokenize(sentence):
        n += 1
        if token in embed:
            vec += embed[token]
        else:
            vec += average_vec
    return vec/n


def getSimilarity(s1, s2):
    vec1 = getEmbedding(s1)
    vec2 = getEmbedding(s2)
    return np.dot(vec1,vec2)/np.linalg.norm(vec1)/np.linalg.norm(vec2)



def getKGE(context, question):
    '''
    Get knowledge graph embedding from context
    choose top five entity
    concatenate those five vectors
    '''
    model, num_node, id_to_node, node_to_id = buildGraph(context)
    vec = torch.zeros(5*300)
    if model is None:
        return vec
    entity_list = node_to_id.keys()
    if len(entity_list) > 5:
        entity_list = sorted(entity_list,key=lambda entity: getSimilarity(entity, question), reverse=True)[:5]
    i = 0
    for entity in entity_list:
        entity_id = node_to_id[entity]
        v = torch.tensor(model.wv[str(entity_id)])
        vec[300*i:300*i+300] += v
        i += 1
    return vec



