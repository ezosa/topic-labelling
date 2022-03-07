# ADAPTED FROM https://github.com/areejokaili/topic_labelling/blob/master/compute_bertscore.py
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 16:01:15 2020
@author: areej
code compute the similarity between references and multiple candidates using BERTScore
"""

from argparse import ArgumentParser
from bert_score import BERTScorer
import numpy as np


def get_bert_score(scorer, multi_preds, multi_golds, num_labels=5):
	P_mult_score, R_mult_score, F1_mult_score = [], [], []
	for num in range(num_labels):
		P, R, F1 = [], [], []
		for ind in range(len(multi_preds)):
			preds = multi_preds[ind][:num+1]
			golds = multi_golds[ind]
			f, p, r = None, None, None
			for pp in preds:
				P_temp, R_temp, F1_temp = scorer.score([pp], [golds])
				if f is None:
					f = F1_temp
					r = R_temp
					p = P_temp
				elif F1_temp > f:
					f = F1_temp
					r = R_temp
					p = P_temp
			P.append(p)
			R.append(r)
			F1.append(f)
		P_mult_score.append(P)
		R_mult_score.append(R)
		F1_mult_score.append(F1)
	return np.asarray(P_mult_score), np.asarray(R_mult_score), np.asarray(F1_mult_score)


# args
parser = ArgumentParser()
# general params
parser.add_argument("-g", "--gold_path", default="examples/gold_labels.txt")
parser.add_argument("-p", "--pred_path", default="examples/output.txt")
parser.add_argument("-l", "--lang", default='fi')
args = parser.parse_args()


print("\n" + "-"*10, "Computing BERTScore", "-"*10)
print("gold:", args.gold_path)
print("pred:", args.pred_path)
print("lang:", args.lang)
print("-"*50 + "\n")

with open(args.pred_path) as f:
	try:
		preds = [line.split(',') for line in f]
		preds = [[s.strip() for s in l] for l in preds]
	except:
		preds = [line.strip().split(',') for line in f]

with open(args.gold_path) as f:
	try:
		golds = [line.split(',') for line in f]
		golds = [[s.strip() for s in l] for l in golds]
	except:
		golds = [line.strip().split(',') for line in f]

num_labels = len(preds[0])
scorer = BERTScorer(lang=args.lang)
P, R, F = get_bert_score(scorer, preds, golds, num_labels=num_labels)

# print("P:", P.shape)
# print("R:", R.shape)
# print("F:", F.shape)

for i in range(num_labels):
	print("Top", i+1, "labels:")
	print("Prec =", P[i].mean().item())
	print("Rec =", R[i].mean().item())
	print("F-score =", F[i].mean().item())
	print("-"*20)
