import pandas as pd
from pathlib import Path
import tiktoken
import openai
import asyncio
from random import randint
import pickle
import numpy as np
import jsonlines
import io
from sklearn.metrics import precision_recall_fscore_support



import dataset_util
class DataSet():
    def __init__(self, path : Path, train : bool, dataset : str, ways : int, test_state : str = ''):
        self.train = train
        self.dataset = dataset.lower()
        self.ways = ways
        self.path = path
        self.test_state = None if not any(test_state) else test_state
        self.data = dataset_util.process_dir(self.path, self.dataset)
        self.prompts = None
        self.tokens = 0
        self.prompt_opts = ['custom', 'kortemeyer']
        
    def make_prompts(self, model : str, prompt_style : str, student_a : int = 1, student_b : int = 0, student_c : int = 0, best_ref : int = 0, good_ref : int = 0, min_ref : int = 0):
        prompt_style =  prompt_style.lower()
        prompt_opts = ['custom', 'kortemeyer']
        ValueError(f'"{prompt_style}" is not an acceptable prompt_style. Please use one of the following: {", ".join(prompt_opts)}.') if prompt_style not in prompt_opts else None
        student_a, student_b, student_c = (-1,-1,-1) if not self.train else (student_a, student_b, student_c if self.ways == 3 else 0)
        self.model = model # Add a checking function here
        best_ref, good_ref, min_ref = (1,0,0) if self.dataset == 'scientsbank' else (best_ref,good_ref, min_ref if self.ways == 3 else 0)
        self.prompts = dataset_util.get_prompt(self.data, self.ways, self.dataset, model, prompt_style, self.train, student_a, student_b, student_c, best_ref, good_ref, min_ref)
    def count_tokens(self, prompt_style: str, model: str):
        '''
        Returns
        -------
        self.tokens : int
            The number of tokens all of the prompts
        '''
        prompt_style =  prompt_style.lower()
        ValueError(f'"{prompt_style}" is not an acceptable prompt_style. Please use one of the following: {", ".join(self.prompt_opts)}.') if prompt_style not in self.prompt_opts else None
        self.encoding = tiktoken.encoding_for_model(model)
        if not any(self.prompts):
            raise AttributeError(f'The prompts have not been generated - please run DataSet.make_prompts prior to calling this function.')
            return
        self.tokens = 0
        
        for prompt in self.prompts[f'{prompt_style}_prompt'].drop_duplicates().to_list():
            self.tokens += len(self.encoding.encode(prompt['messages'][0]['content'])) + len(self.encoding.encode(prompt['messages'][1]['content'])) +  (len(self.encoding.encode(prompt['messages'][2]['content'])) if self.train else 0)
        
        return self.tokens
    
    def tune_messages(self, prompt_style: str):
        prompt_style =  prompt_style.lower()
        ValueError(f'"{prompt_style}" is not an acceptable prompt_style. Please use one of the following: {", ".join(self.prompt_opts)}.') if prompt_style not in self.prompt_opts else None
        if not any(self.prompts):
            raise AttributeError(f'The prompts have not been generated - please run DataSet.make_prompts prior to calling this function.')
            return
        if not self.train:
            raise AttributeError('This dataset is not meant to be used for fine-tuning. Please use a dataset that is, or reinitialize this dataset to be used for finetuning.')
            return
        messages = self.prompts[f'{prompt_style}_prompt'].drop_duplicates().to_list()

        return messages




    
    async def single_prompt(self, client : openai.AsyncOpenAI, messages, model, prompt_style : str, question_id):
        mask_check = lambda x : x in messages[1]['content']
        mask = self.prompts['answer_id'].apply(mask_check)
        mask2 = self.prompts['question_id'] == question_id



        completion = await client.chat.completions.create(messages=messages, model=model)
        # await asyncio.sleep(randint(6,10))
        # completion = {'hehe' : 'lol get faked'}
        
        self.prompts.loc[mask,f'{prompt_style}_{model}_completion'] = [completion for _ in range(self.prompts.loc[mask].shape[0])]
        

    async def run_prompts(self, client : openai.AsyncOpenAI, model : str, prompt_style : str, rate_limit : int):
        tasks = []
        tokens_used = 0
        for i, q_id in enumerate(self.prompts['question_id'].drop_duplicates().to_list()):
            msgs = self.prompts.loc[self.prompts['question_id'] == q_id][f'{prompt_style}_prompt'].iloc[0]['messages']
            msg_token = len(self.encoding.encode(msgs[0]['content'])) + len(self.encoding.encode(msgs[1]['content']))
            msg_token = msg_token * 1.13
            if msg_token + tokens_used + 3000 >= rate_limit:
                print('Rate limit reached, sleeping for a bit :)')
                await asyncio.sleep(70)
                print('Up and awake')
                tokens_used = 0
            tokens_used += msg_token + 3000
            tasks.append(asyncio.create_task(self.single_prompt(client, msgs, model, prompt_style, q_id)))
            
            print(f"{i} : {tokens_used}")
        for task in tasks:
            await task
            

    async def gpt_async(self, client : openai.AsyncOpenAI, model : str, prompt_style : str, rate_limit : int ,save : bool = True):
        prompt_style =  prompt_style.lower()
        ValueError(f'"{prompt_style}" is not an acceptable prompt_style. Please use one of the following: {", ".join(self.prompt_opts)}.') if prompt_style not in self.prompt_opts else None
        loop = asyncio.get_event_loop()
        pts = loop.create_task(self.run_prompts(client, model, prompt_style, rate_limit))
        await pts

        if save:
            self.write_file(model, prompt_style)
        
        self.process_responses(model, prompt_style)

    
    def process_responses(self, model : str, prompt_style : str):
        main_df = pd.DataFrame()
        for i, resp in enumerate(self.prompts[f'{prompt_style}_{model}_completion'].drop_duplicates().to_list()):
            msg = resp.choices[0].message.content
            if msg.count('\n') > 0 and msg.count(':') > 0:
                data = [li.split(':') for li in msg.split('\n')]
            elif msg.count(':') > 0: 
                data = [li.split(':') for li in msg.replace('\"', '').split(',')]
            else:
                data = [li.split(',') for li in msg.replace('\"', '').split('\n')][1:]
            
            try:
                df = pd.DataFrame(data=data, columns = ['answer_id', f'{prompt_style}_{model}_answer'])
            except Exception as e:
                print(f'{i} : {resp}')
                raise e
                

            df[f'{prompt_style}_{model}_answer'] = df[f'{prompt_style}_{model}_answer'].apply(lambda x : x.strip().lower() if x else x)
            df['answer_id'] = df['answer_id'].apply(lambda x : x.strip() if x else x)
            df[f'{prompt_style}_{model}_answer'] =  df[f'{prompt_style}_{model}_answer'].apply(lambda x : 'incorrect' if x == 'incorr' else x).astype('category')
            
            main_df = pd.concat([main_df, df])

        self.prompts = self.prompts.merge(main_df, on='answer_id', how='left')
        self.prompts[f'{prompt_style}_{model}_correct'] = self.prompts[f'{prompt_style}_{model}_answer'] == self.prompts['accuracy']

    def load_processed(self, model : str, prompt_style : str):
        with open(f'semeval_storage/{"train" if self.train else "test"}/{str(self.ways)}way/{self.dataset}/{self.test_state if not self.train else ""}/{model}_{prompt_style}', mode='rb') as reader:
            modeled_responses = pickle.load(reader)
        self.prompts = self.prompts.merge(modeled_responses[['answer_id', f'{prompt_style}_{model}_completion']], on='answer_id', how='left')

        self.process_responses(model, prompt_style)

    def write_file(self, model : str, prompt_style : str):
        with open(f'semeval_storage/{"train" if self.train else "test"}/{str(self.ways)}way/{self.dataset}/{self.test_state if not self.train else ""}/{model}_{prompt_style}', mode='wb') as writer:
            pickle.dump(self.prompts, writer)

    def model_f1_score(self, model : str, prompt_style : str):
        ground_true = self.prompts['accuracy'].to_numpy(dtype=str, copy=True)
        model_prediction = self.prompts[f'{prompt_style}_{model}_answer'].to_numpy(dtype=str, copy=True)

        precision, recall, f1score, support = precision_recall_fscore_support(ground_true, model_prediction, average='weighted')
        return f1score

    
