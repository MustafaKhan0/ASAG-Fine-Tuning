from pathlib import Path
import xmltodict as xmd
import os
import pandas as pd

def process_xml(path: Path, dataset: str):
    '''
    process_xml : Converts xml file from scientsbank or beetle dataset to a pandas DataFrame
    ----------
    Parameters
    path : pathlib.Path
        The path to the xml file which you want to convert
    dataset : str
        The dataset being processed - either "scientsbank" or "beetle" (otehrwise raises ValueError)

    Returns
    -------
    df : pandas.DataFrame
        The dataframe containing the data from that xml file
    '''

    dataset = dataset.lower()

    if dataset not in ['scientsbank', 'beetle']:
        raise ValueError(f'\"{dataset}\" is not a valid value for \'dataset\' input. Use either \"sciEntsBank\" or \"beetle\".')
    with open(path, 'r') as f:
        data = f.read()
    master = xmd.parse(data)

    q_id = master['question']['@id']; q_text = master['question']['questionText']; q_module =  master['question']['@module']

    
    reference_answers = master['question']['referenceAnswers']['referenceAnswer']

    
    reference_processed = [None, None, None]
    if dataset == 'scientsbank':
        best_answers = [reference_answers]
        good_answers = []
        minimal_answers = []




    elif type(reference_answers) == dict:
        best_answers = [reference_answers] if reference_answers['@category'] == 'BEST' else []
        good_answers = [reference_answers] if reference_answers['@category'] == 'GOOD' else []
        minimal_answers = [reference_answers] if reference_answers['@category'] == 'MINIMAL' else []




    elif type(reference_answers) in [set, list]:
        best_answers = [a for a in reference_answers if a['@category'] == 'BEST']
        good_answers = [a for a in reference_answers if a['@category'] == 'GOOD']
        minimal_answers = [a for a in reference_answers if a['@category'] == 'MINIMAL']

        

    
  
    reference_processed = [best_answers, good_answers, minimal_answers]
    answers = master['question']['studentAnswers']['studentAnswer']
    data = []
    answers = [answers] if type(answers) == dict else answers
    for answer in answers:
        data.append(
            [q_id, q_text, q_module] + reference_processed + [answer['@id'], answer['#text'], answer['@accuracy']]
        )
    
    columns = ['question_id', 'question_text', 'module', 'best_answers', 'good_answers', 'minimal_answers', 'answer_id', 'answer_text', 'accuracy']
    df = pd.DataFrame(data=data, columns=columns)
    return df

    
def process_dir(path : Path, dataset):
    if dataset.lower() not in ['scientsbank', 'beetle']:
        raise ValueError(f'\"{dataset}\" is not a valid value for \'dataset\' input. Use either \"sciEntsBank\" or \"beetle\".')
    files = os.listdir(path)
    dfs = {}
    for file in files:
        dfs[file] = process_xml(path / file, dataset)
    

    joined = pd.concat(dfs.values(), ignore_index=True)
    return joined

def custom_prompt(df : pd.DataFrame, ways : int, dataset : str, student_counts : tuple, ref_counts : tuple, train : bool = False):
    # Many assumptions are made about the DF
    # First, that  ans_df inputed is entirely one question
    student_A_count, student_B_count, student_C_count = student_counts
    best_ref_count, good_ref_count, minimal_ref_count = ref_counts
    checker = lambda a : 1 if any(a) else 0

    grade_scale_dict = {
        2 : 'correct or incorrect',
        3 : 'correct, incorrect, or contradictory; contradictory only in the case that the answer contradicts the provided correct answers'
    }

    grade_sample_dict = {
        2 : 'correct or incorrect',
        3 : 'correct, incorrect, or contradictory'
    }

    count_dict = {
        2 : [student_A_count, student_B_count],
        3 : [student_A_count, student_B_count, student_C_count]
    }

    student_ans_id = {}
    ref_ans_id = {}
    role = 'This asistant is a chatbot designed to assess students\' short answer responses on an exam.'

    best_ref_count += 1; good_ref_count += 1; minimal_ref_count += 1
    best_ref_count = min(best_ref_count, checker(df['best_answers']),  len(df['best_answers'].iloc[0]))
    good_ref_count = min(good_ref_count, checker(df['good_answers']), len(df['good_answers'].iloc[0]))
    minimal_ref_count = min(minimal_ref_count, checker(df['minimal_answers']), len(df['minimal_answers'].iloc[0]))
    if not (any(df['best_answers'].iloc[0])):
        best_ref_count = 0
    if not (any(df['good_answers'].iloc[0])):
        good_ref_count = 0
    if not (any(df['minimal_answers'].iloc[0])):
        minimal_ref_count = 0
    best_answers = df['best_answers'].iloc[0][:best_ref_count]
    good_answers = df['good_answers'].iloc[0][:good_ref_count]
    min_answers = df['minimal_answers'].iloc[0][:minimal_ref_count]

    module_check = lambda module, dataset: f' in {module}' if dataset == 'beetle' else ''
    any_check = lambda ans: ans if any(ans) else ''
    any_sep_check = lambda check, rl: rl if any(check) else ''

    best_answer_str = [f' Best_answer_{str(i+1)} - "{str(ans["#text"])}, "' for i,ans in enumerate(best_answers)]
    best_str = [f' BEST - what the optimal answer would look like: '] + best_answer_str + ['.']
    best_str = any_sep_check(best_answer_str, best_str)
    best_str = ''.join(best_str)

    good_answer_str = [f'Good_answer_{str(i+1)} - "{str(ans["#text"])}, "' for i,ans in enumerate(good_answers)]
    good_str = [f' GOOD - a sufficient answer: '] + good_answer_str + ['.']
    good_str = any_sep_check(good_answer_str, good_str)
    good_str = ''.join(good_str)

    minimal_answer_str = [f'Minimal_answer_{str(i+1)} - "{str(ans["#text"])}, "' for i,ans in enumerate(min_answers)]
    minimal_str = [f' Minimal - answers that are not correct: '] + minimal_answer_str + ['.']
    minimal_str = any_sep_check(minimal_answer_str, minimal_str)
    minimal_str = ''.join(minimal_str)
    
    student_A_ans = df[df['accuracy'] == 'correct']
    student_B_ans = df[df['accuracy'] == 'incorrect']
    student_C_ans = df[df['accuracy'] == 'contradictory']
    #student_A_count = len(student_A_ans) if student_A_count == -1 else student_A_count + 1 
    if student_A_count == -1:
        student_A_count = len(student_A_ans)
    if student_B_count == -1:
        student_B_count = len(student_B_ans)
    if student_C_count == -1:
        student_C_count = len(student_C_ans)
    
    student_A_ans = student_A_ans[:student_A_count]
    student_B_ans = student_B_ans[:student_B_count]
    student_C_ans = student_C_ans[:student_C_count]

    final_sans = pd.concat([student_A_ans, student_B_ans, student_C_ans]).sample(frac=1)
    final_sans['llm_ind'] = list(range(final_sans.shape[0]))
    final_ans_li = []
    for i,ans in final_sans.iterrows(): 
        final_ans_li.append(f'student_answer_{str(ans["llm_ind"] + 1)} - "{ans["answer_text"]}"')
        student_ans_id[ans["llm_ind"] + 1] = ans['answer_id']
    final_ans_li.append('.')
    final_ans_str = ', '.join(final_ans_li)
    final_sam_li = []
    for i,ans in final_sans.iterrows(): 
        final_sam_li.append(f'student_answer_{str(ans["llm_ind"] + 1)} - "{grade_sample_dict[ways]}"')
    final_sam_li.append('.')
    final_sam_str = ', '.join(final_sam_li)
    final_sol_li = []
    for i,ans in final_sans.iterrows(): 
        final_sol_li.append(f'student_answer_{str(ans["llm_ind"] + 1)} - "{ans["accuracy"]}"')
    final_sol_li.append('.')
    final_sol_str = ', '.join(final_sol_li)
    beginning_text = 'Suppose you are an educator, specifically, a K-12 teacher, focusing in science.'
    module_text = f' You are grading an exam which aims to assess students\' understanding{module_check(df["module"].iloc[0], dataset)}.'
    questionText = f' This is the question they have been asked: "{df["question_text"].iloc[0]}". You should assess the student responses on the following scale: {grade_scale_dict[ways]}.'
    ref_intro_text = f' You can gain a better understanding of the task through the following reference responses.'
    ref_mid_text = f' They are classified in the following {any(best_answers) + any(good_answers) + any(min_answers)} category(s): '
    ref_cat_list = ["BEST" if any(best_answers) else "", "GOOD" if any(good_answers) else "", "MINIMAL" if any(min_answers) else ""]
    ref_cat_list = list(filter(None, ref_cat_list))
    ref_cat_text = ' ,'.join(ref_cat_list)
    ref_end_text = f' {ref_cat_text}.{best_str}{good_str}{minimal_str}'
    task_intro_text = f' Based on these reference answers, could you grade the following {sum(count_dict[ways])} student responses.'
    task_mid_text = f' Each number represents a different student\'s response to the same question: {final_ans_str}'
    task_end_text = f' Please respond in the following format: {final_sam_str}'
    
    prompt_text = beginning_text + module_text + questionText + ref_intro_text + ref_mid_text + ref_end_text + task_intro_text + task_mid_text + task_end_text
    answer_text = f'Sure! Here are the grades that these students recieved: {final_sol_str}.'
    prompt_text = prompt_text.replace(r'\\', '', -1); answer_text = answer_text.replace(r'\"', '', -1)

    ref_ans_id = {'rids' : [ans['@id'] for ans in best_answers + good_answers + min_answers]}

    return role, prompt_text, student_ans_id, ref_ans_id, answer_text


def kortemeyer_prompt(df: pd.DataFrame, ways : int, dataset : str, student_counts : tuple, ref_counts : tuple, train : bool):
    # Assumes that the DF contains only one question

    # Uses the prompt used in Kortemeyer et al 23
    # 
    student_A_count, student_B_count, student_C_count = student_counts
    best_ref_count, good_ref_count, minimal_ref_count = ref_counts
    ref_role = f'Role: You are an assistant grading short student answers. You provide your grades solely in a CSV table with the columns ”ID” and ”correctness, where you list the full (unshortened) ID and your grading result.”. You grade based on a reference answer that will be provided. You grade as ”correct” if the student answer is a complete and correct paraphrase of the reference answer. {"You grade as ”contradictory” if the student answer explicitly contradicts the reference answer. " if ways == 3 else ""}You grade as ”incorrect” otherwise.'
    no_ref_role = f'Role: You are an assistant grading short student answers. You provide your grades solely in a CSV table with the columns ”ID” and ”correctness, where you list the full (unshortened) ID and your grading result.”. You grade as ”correct” if the student answer is correct and comprehensive. {"You grade as ”contradictory” if the student answer explicitly contradicts the reference answer. " if ways == 3 else ""}You grade as ”incorrect” otherwise.'
    ref = True if any(df.iloc[0]['best_answers']) else False
    role = ref_role if ref else no_ref_role

    intro_text = f'Grade the student answers to the question'
    question_text = f'{df["question_text"].iloc[0]}'
    ref_text = f'The reference answer is given as "{df["best_answers"].iloc[0][0]["#text"]}". ' if ref else ''
    ref_ans_id = {'rids' : [df['best_answers'].iloc[0][0]['@id']]}
    stud_intro = f'The student answers are listed below in the format ”ID:answer”'
    ans_list = []
    student_ans_ids = {}
    eval_ans = []
    for row_ind, row in df.iterrows():
        student_ans_ids[row_ind + 1] = row["answer_id"]
        ans_list.append(f'{row["answer_id"]}:{row["answer_text"]}')
        eval_ans.append(f'{row["answer_id"]}:{row["accuracy"]}')
    ans_text = '. \n'.join(ans_list)



    prompt_text = f'{intro_text} "{question_text}". {ref_text}{stud_intro}. {ans_text}.'
    answer_text = ','.join(eval_ans)
    

    return role, prompt_text, student_ans_ids, ref_ans_id, answer_text



def get_prompt(df : pd.DataFrame, ways : int, dataset : str, model : str, prompt_style : str, tune : bool = False, student_A_count : int = 1, student_B_count : int =0, student_C_count : int = 0, best_ref_count : int = 0, good_ref_count : int = 0, minimal_ref_count : int = 0):
    ''''
    get_rows - converts a DataFrame into a format for OpenAI finetuning API, writes to jsonl
    ----------
    Parameters
    df : pandas.DataFrame
        The dataframe to be used
    ways : int
        The number of possible outputs for grading (MUST BE 2 OR 3)
    dataset : str
        Which dataset is being used (MUST BE \"scientsbank\" or \"beetle\")
    model : str
        Which model is being used (MUST BE \"gpt-3.5\", or \"gpt4\", or \"davinci-003\")
    filename : str or pathlib.Path
        The path/filename to write the final file to
    tune : bool
        Optional - True if this is to fine-tune, false otherwise. Default is false. Controls where or not there is response from the model. 
    student_A_count : int
        Optional - How many of the correct student answers to use. min==1. - -1 means the max possible.
    student_B_count : int
        Optional - How many of the incorrect student answers to use. -1 means the max possible.
    student_C_count : int
        Optional - How many of the contradictory student answers to use. -1 means the max possible.
    best_ref_count : int
        Optional - How many of the best reference answers to use. 
    good_ref_count : int
        Optional - How many of the good reference answers to use
    minimal_ref_count : int
        Optional - How many of the minimal reference answers to use

    NOTES:
    - student_A, student_B, student_C counts and best_ref, good_ref, and minimal_ref counts, -1 means the max
    
    '''
    dfc = df.copy(deep=True)
    dataset = dataset.lower()
    model = model.lower()

    

    if dataset not in ['scientsbank', 'beetle']:
        raise ValueError(f'\"{dataset}\" is not a valid value for \'dataset\' input. Use either \"sciEntsBank\" or \"beetle\".')
    if ways not in [2,3]:
        raise ValueError(f'\'{str(ways)}\' is not a 2-way or a 3-way classification. Use either the integers 2 or 3.')
    #if model not in ['gpt-3.5', 'gpt4', 'davinci-003']:
    #    raise ValueError(f'\'{model}\' is not a valid model. Use \"gpt-3.5\", \"gpt4\", or \"davinci-003\".')
    
    questions = list(set(dfc['question_id']))

    messages = []
    data = []

    prompt_sty_key = {
        'custom' : custom_prompt, 
        'kortemeyer' : kortemeyer_prompt,
    }

    

    for j,question in enumerate(questions):
        mask = dfc['question_id'] == question
        messages = {}
        answers = dfc[dfc['question_id'] == question]

        role, prompt_text, student_ans_id, ref_ans_id, answer_text = prompt_sty_key[prompt_style](answers, ways, dataset, (student_A_count, student_B_count, student_C_count), (best_ref_count, good_ref_count, minimal_ref_count), tune)
        if model == 'davinci-003':
            if tune:
                messages = {{"prompt" : prompt_text, "completion" : answer_text}}
            else: 
                messages = {{"prompt" : prompt_text}}

        else:
            if tune:
                messages = {
                        "messages" : [
                            {"role" : "system", "content" : role},
                            {"role" : "user", "content" : prompt_text},
                            {"role" : "assistant", "content" : answer_text}
                        ]
                    }
            else:
                messages = {
                        "messages" : [
                            {"role" : "system", "content" : role},
                            {"role" : "user", "content" : prompt_text},
                        ]
                    }
        dfc.loc[mask, f'{prompt_style}_prompt'] = [messages] *  mask.sum() #['messages'][0]['content']

        dfc.loc[mask, 'ref_answers_ids'] =   [ref_ans_id] * mask.sum()
        dfc.loc[mask, 'student_answer_ids'] = [student_ans_id] *  mask.sum()


        
        
        
    
    
    
    # I should probably make a system for storing objects (the dicts) so they aren't volatile - its fucking annoying and I can't recover the model responses
    # dfc['ref_answers_ids'] = dfc['ref_answers_ids']['ids']
    return dfc
    
    