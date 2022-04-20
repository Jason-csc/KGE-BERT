import torch
from torch.utils.data import Dataset

from typing import List
import math
import numpy as np
import json


from kGraph import buildGraph, getKGE
from transformers import AutoTokenizer



class KGEDataset_KG(Dataset):
    """Dataset for QA task on SQuAD2.0"""
    def __init__(self, dataset, kg=None):
        super().__init__()
        self.kg = kg
        self.whole_data = []
        qq = 0
        for d in dataset:
            dt = {}
            dt['input_ids'] = d['input_ids']
            dt['token_type_ids'] = d['token_type_ids']
            dt['attention_mask'] = d['attention_mask']
            dt['start_positions'] = d['start_positions']
            dt['end_positions'] = d['end_positions']
            context = d["context"]
            question = d["question"]
            if kg is None:
                dt["KG"] = getKGE(context,question)
            self.whole_data.append(dt)
            qq += 1
            if qq%10000 == 0:
                print(qq,len(dataset))

    
    def __len__(self):
        return len(self.whole_data)

    def __getitem__(self, index):
        if not self.kg is None:
            newdata = self.whole_data[index]
            newdata["KG"] = self.kg[index]
            return newdata
        return self.whole_data[index]




class KGEDataset(Dataset):
    """Dataset for QA task on SQuAD2.0"""
    def __init__(self, dataset):
        super().__init__()
        self.whole_data = []
        qq = 0
        for d in dataset:
            dt = {}
            dt['input_ids'] = d['input_ids']
            dt['token_type_ids'] = d['token_type_ids']
            dt['attention_mask'] = d['attention_mask']
            dt['start_positions'] = d['start_positions']
            dt['end_positions'] = d['end_positions']
            context = d["context"]
            question = d["question"]
            dt["KG"] = getKGE(context,question)
            self.whole_data.append(dt)
            qq += 1
            if qq%10000 == 0:
                print(qq,len(dataset))

    
    def __len__(self):
        return len(self.whole_data)

    def __getitem__(self, index):
        return self.whole_data[index]
        




def basic_collate_fn(batch):
    """Collate function for basic setting."""
    input_ids = [torch.tensor([i['input_ids']]) for i in batch]
    input_ids = torch.cat(input_ids,0)
    token_type_ids = [torch.tensor([i['token_type_ids']]) for i in batch]
    token_type_ids = torch.cat(token_type_ids,0)
    attention_mask = [torch.tensor([i['attention_mask']]) for i in batch]
    attention_mask = torch.cat(attention_mask,0)
    kg_embedding = [i['KG'] for i in batch]
    kg_embedding = torch.vstack(kg_embedding)

    start = [i['start_positions'] for i in batch]
    end = [i['end_positions'] for i in batch]

    return input_ids, token_type_ids, attention_mask, kg_embedding, start, end



max_length = 384
stride = 128
model_checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


def preprocess_train(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
    )
    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []
    context_list = []
    question_list = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        context_list.append(examples["context"][sample_idx])
        question_list.append(questions[sample_idx])
        answer = answers[sample_idx]
        if len(answer["answer_start"]) == 0:
            start_positions.append(0)
            end_positions.append(0)
        else:
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label is (0, 0)
            if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)
    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    inputs["context"] = context_list
    inputs["question"] = question_list
    return inputs







def preprocess_val(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
    )
    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []
    context_list = []
    question_list = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        context_list.append(examples["context"][sample_idx])
        question_list.append(questions[sample_idx])
        answer = answers[sample_idx]
        if len(answer["answer_start"]) == 0:
            start_positions.append([0])
            end_positions.append([0])
        else:
            start_tmp = []
            end_tmp = []
            for j in range(len(answer["answer_start"])):
                start_char = answer["answer_start"][j]
                end_char = answer["answer_start"][j] + len(answer["text"][j])
                sequence_ids = inputs.sequence_ids(i)

                # Find the start and end of the context
                idx = 0
                while sequence_ids[idx] != 1:
                    idx += 1
                context_start = idx
                while sequence_ids[idx] == 1:
                    idx += 1
                context_end = idx - 1

                # If the answer is not fully inside the context, label is (0, 0)
                if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
                    start_tmp.append(0)
                    end_tmp.append(0)
                else:
                    # Otherwise it's the start and end token positions
                    idx = context_start
                    while idx <= context_end and offset[idx][0] <= start_char:
                        idx += 1
                    start_tmp.append(idx - 1)

                    idx = context_end
                    while idx >= context_start and offset[idx][1] >= end_char:
                        idx -= 1
                    end_tmp.append(idx + 1)
            start_positions.append(start_tmp)
            end_positions.append(end_tmp)
    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    inputs["context"] = context_list
    inputs["question"] = question_list
    return inputs









