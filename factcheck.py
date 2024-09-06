# factcheck.py

import torch
from typing import List
import numpy as np
import spacy
import gc
import nltk
import math
import json
import ast
import re

nltk.download("stopwords")
nltk.download('punkt')

WR_THRESHOLD = 0.33

ENT_SIM_THRESHOLD = 0.2
ENT_SIM_MIN = 0.2
ENT_TOLERANCE = 0.01
ENT_THRESHOLD_MIN = 0.5
ENT_THRESHOLD_HIGH = 0.7
ENT_THRESHOLD_FULL = 0.95



class FactExample:
    """
    :param fact: A string representing the fact to make a prediction on
    :param passages: List[dict], where each dict has keys "title" and "text". "title" denotes the title of the
    Wikipedia page it was taken from; you generally don't need to use this. "text" is a chunk of text, which may or
    may not align with sensible paragraph or sentence boundaries
    :param label: S, NS, or IR for Supported, Not Supported, or Irrelevant. Note that we will ignore the Irrelevant
    label for prediction, so your model should just predict S or NS, but we leave it here so you can look at the
    raw data.
    """
    def __init__(self, fact: str, passages: List[dict], label: str):
        self.fact = fact
        self.passages = passages
        self.label = label

    def __repr__(self):
        return repr("fact=" + repr(self.fact) + "; label=" + repr(self.label) + "; passages=" + repr(self.passages))


class EntailmentModel:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def check_entailment(self, premise: str, hypothesis: str):
        with torch.no_grad():
            # Tokenize the premise and hypothesis
            inputs = self.tokenizer(premise, hypothesis, return_tensors='pt', truncation=True, padding=True)
            # Get the model's prediction
            outputs = self.model(**inputs)
            prediction = torch.softmax(outputs["logits"][0], -1).tolist()
            # label_names = ["S", "neutral", "NS"]
            # prediction = {name: round(float(pred) * 100, 1) for pred, name in zip(prediction, label_names)}
            logits = outputs.logits

        # Note that the labels are ["entailment", "neutral", "contradiction"]. There are a number of ways to map
        # these logits or probabilities to classification decisions; you'll have to decide how you want to do this.

        # print(prediction)
        # print(logits)

        # To prevent out-of-memory (OOM) issues during autograding, we explicitly delete
        # objects inputs, outputs, logits, and any results that are no longer needed after the computation.
        del inputs, outputs, logits
        gc.collect()

        return prediction


class FactChecker(object):
    """
    Fact checker base type
    """

    def predict(self, fact: str, passages: List[dict]) -> str:
        """
        Makes a prediction on the given sentence
        :param fact: same as FactExample
        :param passages: same as FactExample
        :return: "S" (supported) or "NS" (not supported)
        """
        raise Exception("Don't call me, call my subclasses")


class RandomGuessFactChecker(object):
    def predict(self, fact: str, passages: List[dict]) -> str:
        prediction = np.random.choice(["S", "NS"])
        return prediction


class AlwaysEntailedFactChecker(object):
    def predict(self, fact: str, passages: List[dict]) -> str:
        return "S"


class TextPreprocessor(object):
    stopWords = set(nltk.corpus.stopwords.words("english"))

    @staticmethod
    def process(text: str, is_fact=False):
        stops = TextPreprocessor.stopWords
        ps = nltk.PorterStemmer()
        if not is_fact:
            clean_text = text.replace("<s>", "").replace("</s>", " . ")
            sentences = nltk.tokenize.sent_tokenize(clean_text)
            clean_sentences = []
            for sentence in sentences:
                clean_sent = nltk.tokenize.word_tokenize(sentence)
                clean_sent = [ps.stem(word.lower()) for word in clean_sent if word not in stops and word.isalnum()]
                if clean_sent:
                    clean_sentences.append(clean_sent)
            return clean_sentences, sentences
        else:
            fact_sentence = nltk.tokenize.word_tokenize(text)
            fact_sentence = [ps.stem(word.lower()) for word in fact_sentence if word not in stops and word.isalnum()]
            return [fact_sentence], [text]

    @staticmethod
    def process_basic(text: str):
        stops = TextPreprocessor.stopWords
        ps = nltk.PorterStemmer()
        clean_text = text.replace("<s>", "").replace("</s>", " . ")
        sentences = nltk.tokenize.sent_tokenize(clean_text)
        sent_words = []
        for sent in sentences:
            words = nltk.tokenize.word_tokenize(sent)
            sent_words.append([ps.stem(word.lower()) for word in words if word not in stops and word.isalnum()])
        return sentences, sent_words, clean_text


class TF_IDF(object):
    def __init__(self, sentences, word_set) -> None:
        self.word_count = {}
        self.word_indices = {}
        self.word_set = word_set
        self.total_documents = len(sentences)
        for i, word in enumerate(word_set):
            self.word_indices[word] = i
        for word in word_set:
            self.word_count[word] = 0
            for sent in sentences:
                if word in sent:
                    self.word_count[word] += 1

    def calc(self, sentence):
        tf_idf_vec = np.zeros((len(self.word_set),))
        for word in sentence:
            word_occurrence = self.word_count[word] + 1 if word in self.word_count else 1
            tf = len([token for token in sentence if token == word])/len(sentence)
            idf = np.log(self.total_documents/word_occurrence)
            value = tf*idf
            tf_idf_vec[self.word_indices[word]] = value
        return tf_idf_vec


class WordRecallThresholdFactChecker(object):
    stopWords = set(nltk.corpus.stopwords.words("english"))
    ps = nltk.PorterStemmer()

    def predict(self, fact: str, passages: List[dict], numeric_mode: bool = False, print_mode: bool = False) -> str:
        truth_sentences = []
        passage_sentences = []
        word_set = set()
        best_sim = -1
        best_sent = None

        clean_fact, _ = TextPreprocessor.process(fact)
        fact_sentence = clean_fact[0]

        for word in fact_sentence:
            word_set.add(word)

        for passage in passages:
            clean_sentences, sentences = TextPreprocessor.process(passage['text'])
            truth_sentences.extend(clean_sentences)
            passage_sentences.extend(sentences)

        for sentence in truth_sentences:
            for word in sentence:
                word_set.add(word)

        tf_idf = TF_IDF(truth_sentences, word_set)
        pred_vec = tf_idf.calc(fact_sentence)

        for i, sentence in enumerate(truth_sentences):
            truth_vec = tf_idf.calc(sentence)
            cos_sim = np.dot(pred_vec, truth_vec)/(np.linalg.norm(pred_vec)*np.linalg.norm(truth_vec))
            if cos_sim > best_sim:
                best_sim = cos_sim
                best_sent = passage_sentences[i]

        pred = "S" if best_sim > WR_THRESHOLD else "NS"
        if print_mode:
            print(f"\n\t Check: {fact} | Pred: {pred} |  Sim : {best_sim} | Sentence : {best_sent}")
        return pred if not numeric_mode else str(best_sim)


class EntailmentFactChecker(object):
    def __init__(self, ent_model: EntailmentModel):
        self.ent_model = ent_model
        self.errors = 0
        self.wrt_checker = WordRecallThresholdFactChecker()

    def predict(self, fact: str, passages: List[dict], numeric_mode: bool = False, print_details: bool = False) -> str:

        cos_sim = float(self.wrt_checker.predict(fact, passages, True))
        if cos_sim > WR_THRESHOLD:
            return "S"

        premises = []
        full_text = []
        hypothesis = fact
        e_max = n_max = c_max = 0.0
        e_sent = n_sent = c_sent = None

        for passage in passages:
            sentences, _, clean_text = TextPreprocessor.process_basic(passage['text'])
            premises.extend(sentences)
            full_text.append(clean_text)

        for i, premise in enumerate(premises):
            guess = [0.0, 0.0, 0.0]
            guess = self.ent_model.check_entailment(premise, hypothesis)
            e_max, e_sent = (guess[0], premise) if guess[0] > e_max else (e_max, e_sent)
            n_max, n_sent = (guess[1], premise) if guess[1] > n_max else (n_max, n_sent)
            c_max, c_sent = (guess[2], premise) if guess[2] > c_max else (c_max, c_sent)

        # if abs(e_max-c_max) <= ENT_TOLERANCE and s_max <= ENT_SIM_THRESHOLD:
        #     # If it's too close try checking whole sentence
        #     for premise in full_text:
        #         guess = self.ent_model.check_entailment(premise, hypothesis)
        #         e_max, e_sent = (guess[0], premise) if guess[0] > e_max else (e_max, e_sent)
        #         n_max, n_sent = (guess[1], premise) if guess[1] > n_max else (n_max, n_sent)
        #         c_max, c_sent = (guess[2], premise) if guess[2] > c_max else (c_max, c_sent)

        pred = None
        if e_max > ENT_THRESHOLD_HIGH:
            pred = "S"
        # elif s_max > ENT_SIM_THRESHOLD:
        #     pred = "S"
        elif e_max > ENT_THRESHOLD_MIN and e_max > c_max:
            pred = "S"
        else:
            pred = "NS"

        if print_details:
            self.errors += 1
            print("------------------------------------------------------------")
            print(f"\n\t Fact : {fact} | Prediction : {pred} | Cos_Sim : {cos_sim} | Errors So Far : {self.errors} of 32")
            print(f"\t E_Max : {e_max} | E_Sent : {e_sent}")
            print(f"\t C_Max : {c_max} | C_Sent : {c_sent}")
            for passage in passages:
                _, _, clean_text = TextPreprocessor.process_basic(passage['text'])
                print(f"\nPassage : \n {clean_text}")
            print("------------------------------------------------------------")

        return pred if not numeric_mode else str(e_max)

# OPTIONAL
class DependencyRecallThresholdFactChecker(object):
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

    def predict(self, fact: str, passages: List[dict]) -> str:
        raise Exception("Implement me")

    def get_dependencies(self, sent: str):
        """
        Returns a set of relevant dependencies from sent
        :param sent: The sentence to extract dependencies from
        :param nlp: The spaCy model to run
        :return: A set of dependency relations as tuples (head, label, child) where the head and child are lemmatized
        if they are verbs. This is filtered from the entire set of dependencies to reflect ones that are most
        semantically meaningful for this kind of fact-checking
        """
        # Runs the spaCy tagger
        processed_sent = self.nlp(sent)
        relations = set()
        for token in processed_sent:
            ignore_dep = ['punct', 'ROOT', 'root', 'det', 'case', 'aux', 'auxpass', 'dep', 'cop', 'mark']
            if token.is_punct or token.dep_ in ignore_dep:
                continue
            # Simplify the relation to its basic form (root verb form for verbs)
            head = token.head.lemma_ if token.head.pos_ == 'VERB' else token.head.text
            dependent = token.lemma_ if token.pos_ == 'VERB' else token.text
            relation = (head, token.dep_, dependent)
            relations.add(relation)
        return relations

