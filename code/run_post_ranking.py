import os, json
import torch
import logging
import argparse
from collections import defaultdict


import pytorch_lightning as pl 

pl.seed_everything(42)

from transformers import AutoTokenizer
from model.ranker_deberta import Ranker
from utils import load_json

from run_generation import transform_question

from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()


def load_dataset(data_dir, answer_dir):
    question_file = data_dir
    answer_file   = os.path.join(answer_dir, 'sample_answers.json')

    datasets = {
        'q': load_json(question_file),
        'a': load_json(answer_file),
    }

    examples = []
    for ID, question in datasets['q'].items():
        answer_counts = datasets['a'][ID].items()
        # answer_counts = lemmatize(datasets['a'][ID]).items()
        for answer, count in answer_counts:
            example = (ID, question, answer, count)
            examples.append(example)

    return examples


# def lemmatize(answer_counts):
#     new_answer_counts = defaultdict(int)
#     for answer, count in answer_counts.items():
#         new_answer = wnl.lemmatize(answer)
#         # if answer != new_answer:
#         #     print(f'{answer} -> {new_answer}')
#         new_answer_counts[new_answer] += count

#     return new_answer_counts

def lemmatize(answer_scores):
    new_answer_scores = defaultdict(list)
    for answer, score in answer_scores:
        new_answer = wnl.lemmatize(answer)
        new_answer_scores[new_answer].append(score)

    results = []
    for answer, scores in new_answer_scores.items():
        results.append((answer, sum(scores)/len(scores)))

    return results


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--data_dir", default="", type=str)
    parser.add_argument("--answer_dir", default="", type=str)
    
    args = parser.parse_args()
    # args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    pl.seed_everything(args.seed)


    model = Ranker.from_pretrained(args.model_name_or_path)
    model.cuda()
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    examples = load_dataset(args.data_dir, args.answer_dir)

    print(model.config)


    answers = defaultdict(list)
    sumc, filc = 0, 0
    for i in range(0, len(examples), args.batch_size):
        batch_examples = examples[i:i+args.batch_size]
        batch_encoding = tokenizer(
            text=[transform_question(example[1]) for example in batch_examples],
            text_pair=[example[2] for example in batch_examples],
            return_tensors='pt', padding=True
        )

        batch_encoding = {k:v.cuda() for k,v in batch_encoding.items()}
        # print(batch_encoding['input_ids'].size())
        scores = model.predict(**batch_encoding)
        for example, score in zip(batch_examples, scores):
            
            # answers[example[0]].append((example[2], score)) #  + example[3]/10))
            # if score > 0.5:
            answers[example[0]].append((example[2], score))
            print(example[0], example[1], '[', example[2], ']', score, example[3])
            # else:
                # print(example[0], example[1], '[', example[2], ']', score, example[3])
                # filc += 1

            # sumc += 1

    print(f'{filc}/{sumc}')

    ranked_predicted_dev = defaultdict(list)
    for q in answers:

        answers_q = answers[q]
        ranked_list = sorted(answers_q, key=lambda it: it[1], reverse=True)

        # 排序后，从频次高到低处理，不超过十个答案

        wnl_word_list = list()
        lemmatized_anslist = []
        cnt = 1  # 计数，最多10个

        for ans, count in ranked_list:

            if cnt > 10:
                break
            ans = ans.lower()
            ans = ans.replace('"','')
            ans = ans.replace('answering:','')
            ans = ans.replace("they're ",'')
            ans = ans.replace("new ",'')
            ans = ans.replace("big ",'')
            ans = ans.replace("great ",'good ')
            if ans in ['', "they're", "go"] or len(ans) < 2:
                continue
            # print(ans, count)

            tags = pos_tag(word_tokenize(ans))
            lem = []
            for w, t in tags:
                if t[0].lower() in ['a','n','v']:
                    lem.append(wnl.lemmatize(w, t[0].lower()))
                else:
                    lem.append(wnl.lemmatize(w))
            lem = ' '.join(lem)

            # 若重复的，跳过；若不重复且答案不超过10个，加入
            flg = True
            for x in wnl_word_list:
                if lem in x:
                    flg = False
                    break
            if flg: # max_len_anslist
                lemmatized_anslist.append(ans)
                wnl_word_list.append(lem)
                cnt += 1

        ranked_predicted_dev[q] = lemmatized_anslist

    print('save to', args.answer_dir+'ranked_list_ranked.jsonl')
    with open(args.answer_dir+'ranked_list_ranked.jsonl', 'w') as f:
        for key in ranked_predicted_dev:
            json.dump({key:ranked_predicted_dev[key]}, f)
            f.write('\n')

if __name__ == '__main__':
    main()