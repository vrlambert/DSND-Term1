import argparse
import json
import torch

from model import load_model, predict

parser = argparse.ArgumentParser(description="Predict using a neural network")

parser.add_argument('input', action='store')
parser.add_argument('checkpoint', action='store')
parser.add_argument('--top_k', action='store', type=int, default=1)
parser.add_argument('--category_names', action='store', default='cat_to_name.json')
parser.add_argument('--gpu', action='store_true', default=False)

args = parser.parse_args()

device = torch.device('cuda' if args.gpu else 'cpu')

with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)
 
print('Loading model from {}'.format(args.checkpoint))
model = load_model(args.checkpoint)
idx_to_name = {v: cat_to_name[k] for k, v in model.class_to_idx.items()}
print('Model loaded')

print('Predicting on {}'.format(device))
probs, labs = predict(args.input, model, device, topk=args.top_k)

print('k. | label | probability')
for i, res in enumerate(zip(probs, labs)):
    label = idx_to_name[res[1]]
    print('{i}. | {lab} | {prob:0.3f}'.format(i=i+1, lab=label, prob=res[0]))
