# Inspired by: https://github.com/hendrycks/test/blob/master/evaluate_flan.py
import string
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
from datasets import load_dataset, DatasetDict
import pandas as pd
import random

import os
import json
import torch
import inspect
import tiktoken
import numpy as np
from tqdm import tqdm, trange
from typing import Any, Dict, List, Optional

from datasets import load_dataset
from transformers.utils import cached_file

from llmtuner.data.template import get_template_and_fix_tokenizer
from llmtuner.eval.MMCBtemplate import get_eval_template
from llmtuner.extras.constants import CHOICES, SUBJECTS
from llmtuner.model import dispatch_model, get_eval_args, load_model_and_tokenizer


class MyEvaluator:

    def __init__(self, args: Optional[Dict[str, Any]] = None) -> None:

        self.model_args, self.data_args, self.eval_args, finetuning_args = get_eval_args(args)
        self.model, self.tokenizer = load_model_and_tokenizer(self.model_args, finetuning_args)
        self.tokenizer.padding_side = "left" # avoid overflow issue in batched inference for llama2
        self.model = dispatch_model(self.model)
        self.template = get_template_and_fix_tokenizer(self.data_args.template, self.tokenizer)

        
        self.eval_template = get_eval_template(self.eval_args.lang)


    @torch.inference_mode()
    def batch_inference(self, batch_input: Dict[str, torch.Tensor]) -> List[str]:  
        output_ids = self.model.generate(
        input_ids=batch_input['input_ids'],
        attention_mask=batch_input['attention_mask'],
        max_length=2048,  # Adjust max_length as needed
        do_sample=False,
        temperature=0.001  # Set temperature to 0.0 for deterministic output
    ).tolist()
        #print(len(output_ids[0]))
    # Slice off the input part from each output sequence
        real_output_ids = [output_id[len(batch_input['input_ids'][i]):] for i, output_id in enumerate(output_ids)]
        #print(len(real_output_ids[0]))
    # Decode the real output ids to strings
        output_strs = self.tokenizer.batch_decode(real_output_ids, skip_special_tokens=True)

        return output_strs


    def eval(self) -> None: #MAIN function
        if "token" in inspect.signature(cached_file).parameters:
            kwargs = {"token": self.model_args.hf_hub_token}
        elif "use_auth_token" in inspect.signature(cached_file).parameters: # for transformers==4.31.0
            kwargs = {"use_auth_token": self.model_args.hf_hub_token}

        


        # It's actually just a support set, not train set
        train_json_path = 'data/json_output/train.json'
        test_json_path = 'data/json_output/test.json'


   
        train_dataset = load_dataset('json', data_files=train_json_path)
        test_dataset = load_dataset('json', data_files=test_json_path)

        #train_dataset=load_dataset('csv',data_files='data/combined_output_100.csv')
        #test_dataset=load_dataset('csv',data_files='data/combined_output_100.csv')

        dataset= DatasetDict({
            'train': train_dataset['train'],
            'test': test_dataset['train']
        })

        inputs, outputs, labels = [], [], []


        for i in trange(len(dataset[self.data_args.split]), desc="Formatting batches", position=1, leave=False):
            support_set = dataset["train"].shuffle().select(range(min(self.eval_args.n_shot, len(dataset["train"]))))
            query, resp, history = self.eval_template.format_example(
                target_data=dataset[self.data_args.split][i],
                support_set=support_set,
                #subject_name=categorys[subject]["name"],
                use_history=self.template.use_history
            )

            #print(query,"/n",resp,'/n',history)
            #exit()

            input_ids, _ = self.template.encode_oneturn( 
                tokenizer=self.tokenizer, query=query, resp=resp, history=history
            )
            inputs.append({"input_ids": input_ids, "attention_mask": [1] * len(input_ids)})
            labels.append(resp)

        for i in trange(0, len(inputs), self.eval_args.batch_size, desc="Predicting batches", position=1, leave=False):
            batch_input = self.tokenizer.pad( 
                inputs[i : i + self.eval_args.batch_size], return_attention_mask=True, return_tensors="pt"
            ).to(self.model.device)
            preds = self.batch_inference(batch_input)
            '''
            test=self.normalize_answer(preds)
            print("##################")
            print(labels[0])
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            print(test)
            print("###########")
            exit()
            '''
            outputs.extend(preds)


        
        normalized_outputs = [self.normalize_answer(output) for output in outputs]
        normalized_labels = [self.normalize_answer(label) for label in labels]
        
        matches = []
        meaningless_resp=0
        for output in normalized_outputs:
            output_words = set(output.split())
            match_found = False
            for word in output_words:
                if word in normalized_labels:
                    matches.append(word)  # Append the matching label
                    match_found = True
                    break  # Break the loop once a match is found
            if not match_found:
                matches.append(random.choice(['neutral', 'positive', 'negative']))# Append random label if no label matches
                meaningless_resp+=1
        
                  



        performance= self.calculate_multiclass_metrics(matches,normalized_labels)
        print(performance)
        print("wrong output is:",meaningless_resp)
        self.save_to_json(performance,normalized_outputs,normalized_labels)

    
    def calculate_multiclass_metrics(self,matches, labels):
        # Initialize and fit a label encoder to the unique labels
        encoder = LabelEncoder()
        unique_labels = set(matches + labels)
        encoder.fit(list(unique_labels))

        # Encode matches and labels
        encoded_matches = encoder.transform(matches)
        encoded_labels = encoder.transform(labels)

        # Calculate accuracy
        accuracy = accuracy_score(encoded_labels, encoded_matches)

        # Calculate micro-average metrics
        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(encoded_labels, encoded_matches, average='micro')

        # Calculate macro-average and weighted-average metrics
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(encoded_labels, encoded_matches, average='macro')
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(encoded_labels, encoded_matches, average='weighted')

        # Calculate metrics for each class
        precision, recall, f1, _ = precision_recall_fscore_support(encoded_labels, encoded_matches, average=None, labels=range(len(unique_labels)))
        class_metrics = {label: {'precision': p, 'recall': r, 'f1_score': f} for label, p, r, f in zip(encoder.classes_, precision, recall, f1)}

        # Prepare and return metrics
        overall_metrics = {
            'accuracy': accuracy,
            'precision_micro': precision_micro,
            'recall_micro': recall_micro,
            'f1_score_micro': f1_micro,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_score_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_score_weighted': f1_weighted
        }

        return {'class_metrics': class_metrics, 'overall_metrics': overall_metrics}
            

        



    # adapted the flowing from Squad v1.1 evaluation, without removing the articles.
    def normalize_answer(self,s):
        """Lower text and remove punctuation, and extra whitespace."""

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_punc(lower(s)))


    def save_to_json(self, dict_data, list1, list2, dict_filename='performance.json', list_filename='outputs.json'):
        # Convert model to string for the path
        model_str = str(self.model_args.model_name_or_path)
        prompt_str=str(self.eval_args.lang)
        shot_str=str(self.eval_args.n_shot)
        # Construct the directory path
        directory_path = os.path.join(model_str,'random', prompt_str, shot_str)

        # Create the directory if it doesn't exist
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        # Save the dictionary to its file
        dict_file_path = os.path.join(directory_path, dict_filename)
        with open(dict_file_path, 'w', encoding='utf-8') as file:
            json.dump(dict_data, file, ensure_ascii=False, indent=4)

        # Save the lists to another file
        lists_file_path = os.path.join(directory_path, list_filename)
        with open(lists_file_path, 'w', encoding='utf-8') as file:
            json.dump({"list1": list1, "list2": list2}, file, ensure_ascii=False, indent=4)

        print(f"Dictionary data saved to {dict_file_path}")
        print(f"List data saved to {lists_file_path}")




if __name__ == "__main__":
    evaluator = MyEvaluator()
    evaluator.eval()
