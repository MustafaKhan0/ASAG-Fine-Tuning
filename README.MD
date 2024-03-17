# Finetuning GPT-3.5 on Automatic Short Answer Grading (ASAG)

This is the code for a research paper exploring the effects of fine tuning on GPT-3.5 Turbo's performance on short answer grading.


## Usage

In order to use the files, you will need to unzip the two zip files in datasets/semeval-2013-task7. In addition, you will need to make a .env file in the main directory of the project, with an OpenAI API key, and the id's for your fine tuned models, and the files for training and validation. An example .env file is in example_env.txt 


The main part of the code is in code/prompting.ipynb. It contains all of the prompting and generation code. The other code file serve to assist prompting.ipynb or display the results from it (code/plotting.ipynb).


## Dataset

The SemEval 2013 Task 7 dataset was used. This is the link from which it was obtained, along with the appropriate citations. This dataset is available under the Creative Commons Attribution-Share Alike License (CC-BY-SA) 3.0. 


https://github.com/myrosia/semeval-2013-task7 

Dzikovska, M.O., Nielsen, R., Brew, C., Leacock, C., Giampiccolo, D., Bentivogli, L., Clark, P., Dagan, I. and Dang, H.T. (2013) "SemEval-2013 Task 7: The Joint Student Response Analysis and 8th Recognizing Textual Entailment Challenge". In Proceedings of the 7th International Workshop on Semantic Evaluation (SemEval 2013), in conjunction with the Second Joint Conference on Lexical and Computational Semantcis (*SEM 2013). Association for Computational Linguistics. Atlanta, Georgia, USA. 13-14 June https://aclanthology.org/S13-2045/

Dzikovska, Myroslava & Nielsen, Rodney & Leacock, Claudia. (2015). The joint student response analysis and recognizing textual entailment challenge: making sense of student responses in educational applications. Language Resources and Evaluation. 50. 10.1007/s10579-015-9313-8. https://dl.acm.org/doi/10.1007/s10579-015-9313-8

## License

The code, graphs, and other original work are liscensed under the [Creative Commons Attribution-Share Alike License (CC-BY-SA) 3.0]

Any code, or figures, or processed data that references the original dataset is attributed to that dataset. In addition, it follows the same License as the dataset itself, the CC-BY-SA 3.0. 

(https://creativecommons.org/licenses/by-sa/3.0/deed.en).