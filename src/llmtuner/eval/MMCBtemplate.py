from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Tuple

from llmtuner.extras.constants import CHOICES

if TYPE_CHECKING:
    from datasets import Dataset


@dataclass
class EvalTemplate:

    system: str
  #  choice: str
    answer: str
    prefix: str


    def parse_example(
        self,
        example: Dict[str, str] 
    ) -> Tuple[str, str]:
        query, resp= example['Text'], example['Sentiment']
        return query+self.answer, resp
        



    def format_example(  
        self,
        target_data: Dict[str, str],
        support_set: "Dataset",
        #subject_name: str,
        use_history: bool
    ) -> Tuple[str, str, List[Tuple[str, str]]]:

        #query, resp= target_data['Text'], target_data['Neutral/Change/Sustain']


        query, resp = self.parse_example(target_data)#
        history = [self.parse_example(support_set[k]) for k in range(len(support_set))] 

        if len(history):
            temp = history.pop(0)

            history.insert(0, (self.system + temp[0], temp[1])) 


        else:
            query = self.system + query   

        if not use_history: 
            query = "\n\n".join(["".join(item) for item in history] + [query])
            history = []
        return query.strip(), resp, history


eval_templates: Dict[str, EvalTemplate] = {}


def register_eval_template(
    name: str,
    system: str,
    #choice: str,
    answer: str,
    prefix: str
) -> None:
    code_string = get_code()
    eval_templates[name] = EvalTemplate(
        system=system+code_string,
        #choice=choice,
        answer=answer,
        prefix=prefix
    )


def get_eval_template(name: str) -> EvalTemplate:
    eval_template = eval_templates.get(name, None)
    assert eval_template is not None, "Template {} does not exist.".format(name)
    return eval_template

'''
register_eval_template(
    name="en",
    system="The following are multiple choice questions (with answers) about {subject}.\n\n",
    choice="\n{choice}. {content}",
    answer="\nAnswer: ",
    prefix=" "
)

'''
import json

def get_code():

    with open('data/code.json', 'r') as f:

        data = json.load(f)

    result = ""

    for code, des in data.items():
        result += f"{code}:{des}\n"

    return result


register_eval_template(
    name="COI",
    system="Based on the Motivational Interview transcript between a psychotherapist and a patient with alcohol abuse issues, please identify the patient's valence about changing their behavior (i.e. reducing alcohol abuse ) as either neutral, positive, or negative. Here are the motivational interviewing behavior Code Descriptions\n ",
    #choice="\n{choice}. {content}",
    answer="\n Based on the Motivational Interview record, answer the following Chain-of-Interaction questions.\n Stage1: What are the prime motivation interview behaviors of the doctor and patient? \nStage2: How is the involvement of patient in this interview, Specifically does the patient self-explore,  and does the patient have any emotional expression? \n Stage3: What is the general sentiment of patient language? And What is the patient's valence about reducing the alcohol use?",
    prefix="\n"
)

register_eval_template(
    name="ablation-",
    system="Based on the Motivational Interview transcript between a psychotherapist and a patient with alcohol abuse issues, please identify the patient's valence about changing their behavior (i.e. reducing alcohol abuse ) as either neutral, positive, or negative. Here are the motivational interviewing behavior Code Descriptions\n ",
    answer="\n Based on the Motivational Interview record, answer the following Chain-of-Interaction questions.\n Stage1: What are the prime motivation interview behaviors of the doctor and patient? \nStage2: How is the involvement of patient in this interview, Specifically does the patient self-explore,  and does the patient have any emotional expression? \n Stage3: What is the general sentiment of patient language? And What is the patient's valence about reducing the alcohol use?",
    prefix="\n"
)
