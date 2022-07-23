import numpy as np
import pandas as pd
from pythainlp.tokenize import word_tokenize
from pythainlp.tag import pos_tag
from pythainlp.tag import chunk_parse
from nltk.chunk import conlltags2tree
import svgling
from collections import Counter
import spacy
from spacy import displacy
from pathlib import Path
import pickle
import itertools
import nltk

nltk.download("punkt")
th_scorer = pickle.load(open('thai_scorer.sav', 'rb'))
en_scorer = pickle.load(open('eng_scorer.sav', 'rb'))
thai_df_col=['question_len_in_word', 'question_contain_negation', 'question_who',
       'question_what', 'question_where', 'question_when', 'question_which',
       'question_why', 'question_how', 'question_how m', 'question_Pronoun',
       'question_Adjective', 'question_NP', 'question_S', 'question_PP',
       'question_vague_count', 'context_len_in_word',
       'context_contain_negation', 'context_who', 'context_what',
       'context_where', 'context_when', 'context_which', 'context_why',
       'context_how', 'context_how m', 'context_Pronoun',
       'context_Conjunction', 'context_Proper Noun', 'context_Number',
       'context_Adjective', 'context_Subordinate', 'context_Adverb',
       'context_NP', 'context_S', 'context_PP', 'context_vague_count',
       'answer_len_in_word', 'answer_contain_negation', 'answer_who',
       'answer_what', 'answer_where', 'answer_when', 'answer_which',
       'answer_why', 'answer_how', 'answer_how m', 'answer_NP', 'answer_S',
       'answer_PP', 'answer_vague_count', 'question_Conjunction',
       'question_Adverb', 'question_Number', 'question_Proper Noun',
       'question_Subordinate', 'answer_Proper Noun', 'answer_Number',
       'answer_Pronoun', 'answer_Adjective', 'answer_Adverb',
       'answer_Subordinate', 'answer_Conjunction']
eng_df_col = ['question_len_in_word', 'question_contain_negation', 'question_who',
       'question_what', 'question_where', 'question_when', 'question_which',
       'question_why', 'question_how', 'question_Pronoun',
       'question_Adjective', 'question_NP', 'question_PP',
       'question_vague_count', 'context_len_in_word',
       'context_contain_negation', 'context_who', 'context_what',
       'context_where', 'context_when', 'context_which', 'context_why',
       'context_how', 'context_Adjective', 'context_Conjunction',
       'context_Pronoun', 'context_Subordinate', 'context_Adverb',
       'context_NP', 'context_PP', 'context_vague_count', 'answer_len_in_word',
       'answer_contain_negation', 'answer_who', 'answer_what', 'answer_where',
       'answer_when', 'answer_which', 'answer_why', 'answer_how',
       'answer_Adjective', 'answer_Conjunction', 'answer_NP', 'answer_PP',
       'answer_vague_count', 'question_Subordinate', 'answer_Adverb',
       'question_Conjunction', 'answer_Pronoun', 'question_Proper Noun',
       'answer_Subordinate', 'context_Proper Noun', 'answer_Proper Noun',
       'question_Adverb', 'question_Number', 'context_Number', 'answer_Number',
       'question_how m', 'question_S']

en_drop_columns=['context_who', 'context_what','context_where', 'context_when', 'context_which', 'context_why',
            'context_how', 'answer_who',
            'answer_what', 'answer_where', 'answer_when', 'answer_which',
            'answer_why', 'answer_how', ]

th_drop_columns=en_drop_columns+['answer_how m','context_how m']
spacy_nlp = spacy.load('en_core_web_sm')

def replace_pos_th(pos_elem):
  if pos_elem=="NPRP":
    return "Proper Noun"
  elif pos_elem in ["PPRS","PDMN","PNTR","PREL"]:
    return "Pronoun"
  elif pos_elem in ["ADJ","NONM","VATT","DONM"]:
    return "Adjective"
  elif pos_elem in ["ADV","ADVN","ADVI","ADVP","ADVS"]:
    return "Adverb"
  elif pos_elem in ["JCRG","JCMP"]:
    return "Conjunction"
  elif pos_elem in ["NCNM","DCNM"]:
    return "Number"
  return pos_elem

def count_pos(pos_list):
  return Counter(pos_list)

def check_clear_th(tmp_pos):
  if ("NCMN" not in tmp_pos) and ("PPRS" not in tmp_pos):
    return True
  return (("NCMN" in tmp_pos) or ("PPRS" in tmp_pos)) and (("JSBR" in tmp_pos) or ("RPRE" in tmp_pos))


def chunk_tag_th(txt):
  m = [(w,t) for w,t in pos_tag(word_tokenize(txt), engine= 'perceptron',corpus = 'orchid')]
  tag = chunk_parse(m)
  p = [(w,t,tag[i]) for i,(w,t) in enumerate(m)]
  return p

def count_chunk_th(tag_chunk_list):
  chunk_list=[tag_chunk_elem[2] for tag_chunk_elem in tag_chunk_list]
  new_chunk_list, new_text, tmp_pos = [],[],[]
  tmp_chunk,tmp_text = "",""
  count_vague=0

  for idx,chunk_elem in enumerate(chunk_list):
    if chunk_elem[0]=="B" :
      if tmp_chunk != "":
        new_chunk_list += [tmp_chunk]
        new_text += [tmp_text]
        if not check_clear_th(tmp_pos):
          count_vague+=1

      tmp_chunk, tmp_text, tmp_pos=chunk_elem[2:], tag_chunk_list[idx][0], [tag_chunk_list[idx][1]]

    elif chunk_elem[0]=="O":
      if tmp_chunk != "":
        new_chunk_list += [tmp_chunk]  
        new_text += [tmp_text]
        if not check_clear_th(tmp_pos):
          count_vague+=1

      new_chunk_list += ["O"]
      new_text += [tag_chunk_list[idx][0]]
      tmp_chunk,tmp_text="",""
      tmp_pos=[]
    else:
      tmp_text+=tag_chunk_list[idx][0]
      tmp_pos+=[tag_chunk_list[idx][1]]

  if tmp_chunk!="":
    new_chunk_list+=[tmp_chunk]
    new_text += [tmp_text]
  
  counter = Counter(new_chunk_list)
  for idx,chunk_elem in enumerate(new_chunk_list):
    if chunk_elem=="S" and len(new_chunk_list)!=1:
      sub_counter=count_chunk_th(chunk_tag_th(new_text[idx]))
      counter+=sub_counter[0]
      count_vague+=sub_counter[1]

  return counter,count_vague


def grammartical_feature_th(txt):

    p = chunk_tag_th(txt)

    pos = [replace_pos_th(pp[1]) for pp in p]
    num_pos=count_pos(pos)
    feat_dict={}

    for k,v in num_pos.items():
      if k in ["Proper Noun","Pronoun","Adjective","Adverb","Number"]:
        feat_dict[k]=v
      elif k in ["Conjunction"]:
        feat_dict[k]=v+num_pos["JSBR"]
      elif k=="JSBR":
        feat_dict["Subordinate"]=v
  
    count_chunk,count_vague=count_chunk_th(p)
    feat_dict["NP"]=count_chunk["NP"]
    feat_dict["S"]=count_chunk["S"]
    feat_dict["PP"]=count_chunk["PP"]
    feat_dict["vague_count"]=count_vague
    return feat_dict
def get_pps(doc):
  "Function to get PPs from a parsed document."
  pps = []
  for token in doc:
      # Try this with other parts of speech for different subtrees.
    if token.pos_ == 'ADP':
      pp = ' '.join([tok.orth_ for tok in token.subtree])
      pps.append(pp)
  return pps
def replace_pos_en(pos_elem):
  if pos_elem=="PROPN":
    return "Proper Noun"
  elif pos_elem=="PRON":
    return "Pronoun"
  elif pos_elem=="ADJ":
    return "Adjective"
  elif pos_elem=="ADV":
    return "Adverb"
  elif pos_elem in ["CONJ","CCONJ"]:
    return "Conjunction"
  elif pos_elem=="NUM":
    return "Number"
  return pos_elem

def check_clear_en(tmp_pos):
  if ("NOUN" not in tmp_pos) and ("PRON" not in tmp_pos):
    return True
  return (("NOUN" in tmp_pos) or ("PRON" in tmp_pos)) and (("SCONJ" in tmp_pos) or ("RPRE" in tmp_pos))


def count_chunk_en(doc):

  count_vague =0
  for i in doc.noun_chunks:
    if not check_clear_en([t.pos_ for t in i]):
      count_vague+=1

  return {"NP":len(list(doc.noun_chunks)),"PP":len(get_pps(doc))},count_vague


def grammartical_feature_en(txt):

    doc=spacy_nlp(txt)

    pos = [replace_pos_en(pp.pos_) for pp in doc]
    num_pos=count_pos(pos)
    feat_dict={}
    for k,v in num_pos.items():
      if k in ["Proper Noun","Pronoun","Adjective","Adverb","Number"]:
        feat_dict[k]=v
      elif k in ["Conjunction"]:
        feat_dict[k]=v+num_pos["SCONJ"]
      elif k=="SCONJ":
        feat_dict["Subordinate"]=v
  
    count_chunk,count_vague=count_chunk_en(doc)
    feat_dict["NP"]=count_chunk["NP"]
    feat_dict["PP"]=count_chunk["PP"]
    feat_dict["vague_count"]=count_vague
    
    return feat_dict


def len_in_word(text,is_th):
  text=text.lower()
  if is_th:
    return len(word_tokenize(text))
  else :
    return len(nltk.word_tokenize(text))

def contain_negation(text,is_th):
  text=text.lower()
  if is_th:
    return "ไม่" in text
  else :
    return ("no" in text) or ("not" in text) or ("never" in text) or ("neither" in text)

def contain_wh(text,is_th):
  text=text.lower()
  if is_th:
    ans = {}
    q_type = ["who","what","where","when","which","why","how","how m"]
    for t in q_type:
      ans[t]=0
    for idx,subs_list in enumerate([["ใคร","ผู้ใด"],["อะไร","สิ่งใด","ว่าอย่างไร","ว่ายังไง"],["ที่ไหน","ที่ใด","เมืองไหน","เมืองใด","เมืองอะไร","ประเทศไหน","ประเทศใด","ประเทศอะไร","จังหวัดไหน","จังหวัดใด","จังหวัดอะไร","อำเภอไหน","อำเภอใด","อำเภออะไร"],["เมื่อไหร่","เมื่อไร","เมื่อใด","ปีไหน","ปีใด","วันที่เท่าไหร่","วันใด","วันไหน","วันที่เท่าไร"],["อันไหน","อันใด"],["ทำไม","เหตุใด","เพราะอะไร"],["อย่างไร"],["กี่","แค่ไหน","เท่าไหร่"]]):
      if any(map(text.__contains__, subs_list)):
        ans[q_type[idx]]=1
        return ans
    return ans
  else :
    ans = {}
    q_type = ["who","what","where","when","which","why","how"]
    for t in q_type:
      ans[t]=0
    first_idx = [text.find(subs) for subs in ["who","what","where","when","which","why","how"]]
    first_idx = [first_idx[idx] if first_idx[idx]!=-1 else len(text) for idx in range(len(first_idx))]
    ans[q_type[np.argmin(first_idx)]]=1
    return ans

def grammartical_feature(text,is_th):
  if is_th:
    return grammartical_feature_th(text)
  else :
    return grammartical_feature_en(text)

def feature_extract_single_text(text,is_th):
  feat_dict={}
  feat_dict["len_in_word"]=len_in_word(text,is_th)
  feat_dict["contain_negation"]=contain_negation(text,is_th)
  feat_dict.update(contain_wh(text,is_th))
  grammar_feat_dict=grammartical_feature(text,is_th)
  feat_dict.update(grammar_feat_dict)

  return feat_dict

def ranking_feature_extract(context,question_list,answer,is_th):

    feat_list = []
    
    a={"answer_"+k:v for k,v in feature_extract_single_text(answer,is_th).items()}
    c={"context_"+k:v for k,v in feature_extract_single_text(context,is_th).items()}
    for question in question_list:
        q={"question_"+k:v for k,v in feature_extract_single_text(question,is_th).items()}
        q.update(c)
        q.update(a)
        feat_list.append(q)
    return feat_list

def get_ranking_score(context, question_list, answer, is_th):


    feat_list=ranking_feature_extract(context,question_list,answer,is_th)
    
    if is_th:
        thai_df=pd.DataFrame(columns=thai_df_col)
        thai_df = thai_df.append(feat_list, ignore_index=True, sort=False).fillna(0)
        thai_df=thai_df.drop(columns=th_drop_columns)

        return [[i,j] for i,j in zip(itertools.chain(*th_scorer.predict(thai_df)),question_list)]
    else:
        eng_df=pd.DataFrame(columns=eng_df_col)
        eng_df = eng_df.append(feat_list, ignore_index=True, sort=False).fillna(0)
        eng_df=eng_df.drop(columns=en_drop_columns)
        return [[i,j] for i,j in zip(itertools.chain(*en_scorer.predict(eng_df)),question_list)]
