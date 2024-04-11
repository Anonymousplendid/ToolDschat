# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

import os
# DeepSpeed Team
from datasets import load_dataset, load_from_disk
from torch.utils.data import Subset
import re
import random
from copy import deepcopy

# The template prompt dataset class that all new dataset porting needs to
# follow in order to have a unified API and unified data format.
class PromptRawDataset(object):

    def __init__(self, output_path, seed, local_rank, dataset_name):
        self.output_path = output_path
        self.seed = seed
        self.local_rank = local_rank
        if os.path.exists(dataset_name):
            self.raw_datasets = load_from_disk(dataset_name)
        elif not dataset_name == 'local/jsonfile':
            self.raw_datasets = load_dataset(dataset_name)

    def get_train_data(self):
        return

    def get_eval_data(self):
        return

    # The prompt should be in the format of: " Human: " + actual_prompt_sentence + " Assistant:"
    def get_prompt(self, sample):
        return

    # The chosen response should be in the format of: " " + actual_response_sentence
    def get_chosen(self, sample):
        return

    # The rejected response should be in the format of: " " + actual_response_sentence
    # If the dataset does not have rejected response, return None
    def get_rejected(self, sample):
        return

    def get_prompt_and_chosen(self, sample):
        return

    def get_prompt_and_rejected(self, sample):
        return


fewshots = [
    {
        "question": "Human: I planted some sunflowers and they are already drooping a bit. What did I do wrong?",
        "answer": """Response 1: Sunflowers are sun-loving flowers, and a lot of people think that they\u2019re best grown in warm weather climates.  It can be done in northern areas, and sometimes they take a while to get established, but they do like full sun to bloom.  So if your plants are sun-loving plants, they probably want to grow in more sun and less shade, and more warm weather.
Response 2: You may have planted them in the wrong kind of soil.  Sunflowers grow best in sandy soils, rather than the clay you probably have.

Evaluation: In this conversation, the response should aim to troubleshoot the drooping sunflowers. Response 1 provides actionable advice on sunlight and warmth requirements, making it helpful and practical. Response 2 prematurely assumes soil issues, lessening its helpfulness. Therefore, the preferred response is Response 1.""",
    },
    {
        "question": "Human: What kind of Christmas decorations can I make out of felt?\n\nAssistant: Felt isn\u2019t a really common material to use for Christmas decorations, but you could probably just buy some felt from the craft store.  You could use it to make your own Christmas tree decoration or paper chains or some other ornament.  You might also be able to use it to make felt ornaments, but if you want to do that you would want to coat them in some water-based material to make them less fragile.  My guess is that you\u2019d also want to give your decorations as gifts, which is probably not a priority for most people these days.\n\nHuman: I think it is pretty common.\n\nAssistant: Oh, I guess it is. Well if you\u2019re going to make felt Christmas decorations for gifts, I recommend just buying them instead of making them. You\u2019ll get better looking decorations.\n\nHuman: I want to have fun making them.\n\nAssistant: You could make some in the shape of stockings or socks.\n\nHuman: Sounds good. Any other ideas?",
        "answer": """Response 1: I don\u2019t think so, though I\u2019m no expert on felt crafts.
Response 2: I don\u2019t know. Maybe a snowflake, a reindeer?

Evaluation: In this conversation, the response should aim to provide creative suggestions for felt Christmas decorations. Response 1 is dismissive and lacks helpfulness, while Response 2 offers creative ideas, aligning with helpfulness and practicality. Therefore, the preferred response is Response 2.""",
    },
    {
        "question": "Human: Are older cars less safe to drive?\n\nAssistant: Older cars may not be as safe as newer ones, although they are often maintained better.  Safety features are added to vehicles from model year to model year, and the quality of the vehicles themselves tends to improve over the years as well.  But many older cars were made before the safety standards were as stringent as they are today, and they may not be as crash-proof.\n\nHuman: Are air bags a lot safer to have than not?",
        "answer": """Response 1: Air bags are a safety feature added to many vehicles starting in the early 1990s.  They have helped make cars safer, and they also make the cars themselves safer, since they\u2019ve helped to reduce the number of crashes the vehicles are involved in.
Response 2: Air bags are designed to help protect the passengers in a car in the event of a crash.  If the air bags work correctly, they can help reduce the risk of injury to the driver and passenger.  They are standard on all new vehicles, and are also available as an option on many older vehicles.  So having air bags can help you be safer on the road, and also in the event of a crash.

Evaluation: In this conversation, the response should aim to inform about vehicle safety. Response 1 inaccurately implies air bags reduce the number of crashes, confusing their purpose, which impacts the honesty and sophistication of the information. Response 2 correctly explains the function of air bags and their impact on safety, providing a more comprehensive and practical answer. Therefore, the preferred response is Response 2.""",
    },
    {
        "question": "Human: How do I make a wreath?\n\nAssistant: To make a wreath, start by gathering materials. You will need green and/or artificial foliage, artificial berries, wooden skewers, wire, and glue. Lay out the foliage in a circle, then place the wooden skewers in the center of the circle. String the wire through the holes in the skewers, then weave the foliage around the wires to form a wreath shape. Once your wreath is finished, add a bow and glue on your artificial berries to create a beautiful holiday decoration.\n\nHuman: What kind of wire do I use?",
        "answer": """Response 1: To make a wreath, you will need wire that is at least 3\u20448 inch in diameter, which is also known as 18-gauge wire. You can buy wire at most hardware stores or craft stores. In a pinch, you can also use string or wire hangers.
Response 2: I would use wire that is made of a material known as \u201cmetal\u201d. Metal wire has the qualities of strength and conductivity, and can be made from materials such as steel, copper, silver, gold, platinum, aluminum, zinc, and iron.

Evaluation: In this conversation, the response should aim to provide specific guidance on choosing wire for wreath-making. Response 1 gives specific and actionable advice for selecting wire, aligning with helpfulness and practicality. Response 2 is too general and not tailored to wreath-making, lacking practicality. Therefore, the preferred response is Response 1.""",
    },
    {
        "question": 'Human: I\'m going to Italy and need to learn some of the language.\n\nAssistant: I\'m happy to help!  How would you like to approach learning?\n\nHuman: I would like to learn some weather words. What is "rain" in Italian?\n\nAssistant: It\'s pioggia. And "snow" is neve.\n\nHuman: Perfect! What about "thunder" and "lightning"?',
        "answer": """Response 1: That's tuono and fulmine.
Response 2: Thunder is tuono. Lightning is fulmine. And thunder in Italian sounds like \"ta-rump-ta-rump\", and lightning like \"car

Evaluation: In this conversation, the response should aim to offer Italian translations for weather terms. Response 1 is clear and accurate, meeting the goal of helpfulness. Response 2 adds unnecessary detail that may confuse. Therefore, the preferred response is Response 1.""",
    },
    {
        "question": "Human: What are some festivals I can go to like Woodstock?\n\nAssistant: Woodstock is a festival that took place on a farm in New York in 1969.\n\nHuman: Right. So it's not around anymore. I just want similar festivals.",
        "answer": """Response 1: There are similar music festivals in a number of places, like Boston, Colorado, France, Spain, London, Canada, Australia, and many more.
Response 2: One festival that comes to mind is Burning Man, which is a music and art festival that happens every year in Black Rock City, Nevada.

Evaluation: In this conversation, the response should aim to suggest festivals akin to Woodstock. Response 1 is vague, while Response 2 provides a specific festival, enhancing its usefulness. Therefore, the preferred response is Response 2.""",
    },
]

prompt = """Here's a dialogue between a human and an AI assistant. The AI's mission is to provide accurate, timely, and relevant information, and to understand and respond to inquiries with detailed, reliable, and neutral information. Evaluate the AI's responses with respect to the following principles and specific criteria:

Helpfulness: The response should directly address the user's question or stated need.
Politeness: The response should maintain a respectful and considerate tone throughout.
Honesty: The response must be truthful and transparent, avoiding misleading information.
Sophistication: The response should demonstrate a depth of knowledge and articulate complex ideas clearly when necessary.
Emotional Awareness: The response should be sensitive to the emotional tone of the human and respond appropriately.
Humility: The response should acknowledge the limitations of the AI's knowledge and capabilities when applicable.
Practicality: The response should provide actionable information or direct the user to where they can find the assistance they need.

Please evaluate each of the AI assistant's responses in this dialogue, considering how well it achieves these objectives. Provide a brief justification for your evaluation of each response."""


def get_prompt():
    fewshots_now = deepcopy(fewshots)
    random.shuffle(fewshots_now)
    fewshots_now = fewshots_now[:5]
    ret = prompt + "\n\n"
    for fewshot in fewshots_now:
        ret += (
            "---\n-- dialogue begin --\n"
            + fewshot["question"]
            + "\n"
            + "-- dialogue end --\n\n"
        )
        ret += fewshot["answer"] + "\n\n\n"
    return ret
        
def get_sample(prompt, choose, reject):
    ret = get_prompt()
    random_float = random.random()
    if random_float < 0.5:
        first = choose.strip()
        second = reject.strip()
        target = 0
    else:
        first = reject.strip()
        second = choose.strip()
        target = 1
    ret += "---\n-- dialogue begin --\n"
    ret += prompt.strip() + "\n"
    ret += "-- dialogue end --\n\n"
    ret += "Response 1: " + first + "\n"
    ret += "Response 2: " + second + "\n\n"
    ret += "Evaluation: In this conversation, the response should aim to"
    return ret, target
    
# English dataset
class DahoasRmstaticDataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "Dahoas/rm-static"
        self.dataset_name_clean = "Dahoas_rm_static"
        
    def get_rlreward(self, sample):
        prompt = sample['prompt'].strip().rstrip("Assistant:").strip()
        choose = sample['chosen'].strip()
        reject = sample['rejected'].strip()
        return get_sample(prompt, choose, reject)

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return sample['prompt']

    def get_chosen(self, sample):
        return sample['chosen']

    def get_rejected(self, sample):
        return sample['rejected']

    def get_prompt_and_chosen(self, sample):
        return sample['prompt'] + sample['chosen']

    def get_prompt_and_rejected(self, sample):
        return sample['prompt'] + sample['rejected']


# English dataset
class DahoasFullhhrlhfDataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "Dahoas/full-hh-rlhf"
        self.dataset_name_clean = "Dahoas_full_hh_rlhf"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return sample['prompt']

    def get_chosen(self, sample):
        return sample['chosen']

    def get_rejected(self, sample):
        return sample['rejected']

    def get_prompt_and_chosen(self, sample):
        return sample['prompt'] + sample['chosen']

    def get_prompt_and_rejected(self, sample):
        return sample['prompt'] + sample['rejected']


# English dataset
class DahoasSyntheticinstructgptjpairwiseDataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "Dahoas/synthetic-instruct-gptj-pairwise"
        self.dataset_name_clean = "Dahoas_synthetic_instruct_gptj_pairwise"

    def get_train_data(self):
        from .data_utils import get_raw_dataset_split_index
        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(self.local_rank, self.output_path,
                                            self.dataset_name_clean,
                                            self.seed, "train_eval", "9,1", 0,
                                            len(dataset))
        dataset = Subset(dataset, index)
        return dataset

    def get_eval_data(self):
        from .data_utils import get_raw_dataset_split_index
        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(self.local_rank, self.output_path,
                                            self.dataset_name_clean,
                                            self.seed, "train_eval", "9,1", 1,
                                            len(dataset))
        dataset = Subset(dataset, index)
        return dataset

    def get_prompt(self, sample):
        return " Human: " + sample['prompt'] + " Assistant:"

    def get_chosen(self, sample):
        return " " + sample['chosen']

    def get_rejected(self, sample):
        return " " + sample['rejected']

    def get_prompt_and_chosen(self, sample):
        return " Human: " + sample['prompt'] + " Assistant: " + sample['chosen']

    def get_prompt_and_rejected(self, sample):
        return " Human: " + sample['prompt'] + " Assistant: " + sample[
            'rejected']


# English dataset
class YitingxieRlhfrewarddatasetsDataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "yitingxie/rlhf-reward-datasets"
        self.dataset_name_clean = "yitingxie_rlhf_reward_datasets"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return sample['prompt'] + "Assistant:"

    def get_chosen(self, sample):
        return sample['chosen'].split("Assistant:")[-1]

    def get_rejected(self, sample):
        return sample['rejected'].split("Assistant:")[-1]

    def get_prompt_and_chosen(self, sample):
        return sample['prompt'] + sample['chosen']

    def get_prompt_and_rejected(self, sample):
        return sample['prompt'] + sample['rejected']


# English dataset
class OpenaiWebgptcomparisonsDataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "openai/webgpt_comparisons"
        self.dataset_name_clean = "openai_webgpt_comparisons"

    def get_train_data(self):
        from .data_utils import get_raw_dataset_split_index
        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(self.local_rank, self.output_path,
                                            self.dataset_name_clean,
                                            self.seed, "train_eval", "9,1", 0,
                                            len(dataset))
        dataset = Subset(dataset, index)
        return dataset

    def get_eval_data(self):
        from .data_utils import get_raw_dataset_split_index
        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(self.local_rank, self.output_path,
                                            self.dataset_name_clean,
                                            self.seed, "train_eval", "9,1", 1,
                                            len(dataset))
        dataset = Subset(dataset, index)
        return dataset

    def get_prompt(self, sample):
        return " Human: " + sample['question']['full_text'] + " Assistant:"

    def get_chosen(self, sample):
        if float(sample['score_0']) >= float(sample['score_1']):
            response = sample['answer_0']
        else:
            response = sample['answer_1']
        # This data has citation square brackets and numbers (e.g., "[1]").
        # Right now we are not doing browser-assisted finetuning, thus we
        # remove these citations to avoid confusing the model.
        response = re.sub(r" [\(\[].*?[\)\]]", "", response)
        response = re.sub(r"[\(\[].*?[\)\]]", "", response)
        return " " + response

    def get_rejected(self, sample):
        if float(sample['score_0']) < float(sample['score_1']):
            response = sample['answer_0']
        else:
            response = sample['answer_1']
        response = re.sub(r" [\(\[].*?[\)\]]", "", response)
        response = re.sub(r"[\(\[].*?[\)\]]", "", response)
        return " " + response

    def get_prompt_and_chosen(self, sample):
        if float(sample['score_0']) >= float(sample['score_1']):
            response = sample['answer_0']
        else:
            response = sample['answer_1']
        response = re.sub(r" [\(\[].*?[\)\]]", "", response)
        response = re.sub(r"[\(\[].*?[\)\]]", "", response)
        return " Human: " + sample['question'][
            'full_text'] + " Assistant: " + response

    def get_prompt_and_rejected(self, sample):
        if float(sample['score_0']) < float(sample['score_1']):
            response = sample['answer_0']
        else:
            response = sample['answer_1']
        response = re.sub(r" [\(\[].*?[\)\]]", "", response)
        response = re.sub(r"[\(\[].*?[\)\]]", "", response)
        return " Human: " + sample['question'][
            'full_text'] + " Assistant: " + response


# English dataset
class StanfordnlpSHPDataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "stanfordnlp/SHP"
        self.dataset_name_clean = "stanfordnlp_SHP"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["validation"]

    def get_prompt(self, sample):
        return " Human: " + sample['history'] + " Assistant:"

    def get_chosen(self, sample):
        if int(sample["labels"]) == 1:
            response = sample["human_ref_A"]
        else:
            response = sample["human_ref_B"]
        return " " + response

    def get_rejected(self, sample):
        if int(sample["labels"]) == 1:
            response = sample["human_ref_B"]
        else:
            response = sample["human_ref_A"]
        return " " + response

    def get_prompt_and_chosen(self, sample):
        if int(sample["labels"]) == 1:
            response = sample["human_ref_A"]
        else:
            response = sample["human_ref_B"]
        return " Human: " + sample['history'] + " Assistant: " + response

    def get_prompt_and_rejected(self, sample):
        if int(sample["labels"]) == 1:
            response = sample["human_ref_B"]
        else:
            response = sample["human_ref_A"]
        return " Human: " + sample['history'] + " Assistant: " + response


# English dataset
class PvduySharegptalpacaoavicunaformatDataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "pvduy/sharegpt_alpaca_oa_vicuna_format"
        self.dataset_name_clean = "pvduy_sharegpt_alpaca_oa_vicuna_format"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        if sample['prompt'] is not None and len(sample['prompt']) > 0:
            return sample['prompt'].replace("USER", "Human").replace(
                "ASSISTANT", "Assistant")
        return None

    def get_chosen(self, sample):
        if sample['label'] is not None and len(sample['label']) > 0:
            return " " + sample['label']
        return None

    def get_rejected(self, sample):
        print(
            f"Warning: dataset {self.dataset_name} does not include rejected response."
        )
        return None

    def get_prompt_and_chosen(self, sample):
        if sample['prompt'] is not None and sample['label'] is not None and len(
                sample['prompt']) > 0 and len(sample['label']) > 0:
            return sample['prompt'].replace("USER", "Human").replace(
                "ASSISTANT", "Assistant") + " " + sample['label']
        return None

    def get_prompt_and_rejected(self, sample):
        print(
            f"Warning: dataset {self.dataset_name} does not include rejected response."
        )
        return None


class LocalJsonFileDataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name, chat_path):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "local/jsonfile"
        self.dataset_name_clean = "jsonfile"
        self.raw_datasets = load_dataset('json',
                                         data_files={
                                             "train":
                                             chat_path + '/data/train.json',
                                             "eval":
                                             chat_path + '/data/eval.json'
                                         })

    def get_train_data(self):
        if self.raw_datasets['train'] is not None:
            return self.raw_datasets['train']
        return None

    def get_eval_data(self):
        if self.raw_datasets['eval'] is not None:
            return self.raw_datasets['eval']
        return None

    # The prompt should be in the format of: " Human: " + actual_prompt_sentence + " Assistant:"
    def get_prompt(self, sample):
        if sample['prompt'] is not None:
            return " " + sample['prompt']
        return None

    # The chosen response should be in the format of: " " + actual_response_sentence
    def get_chosen(self, sample):
        if sample['chosen'] is not None:
            return " " + sample['chosen']
        return None

    # The rejected response should be in the format of: " " + actual_response_sentence
    # If the dataset does not have rejected response, return None
    def get_rejected(self, sample):
        if sample['rejected'] is not None:
            return " " + sample['rejected']
        return None

    def get_prompt_and_chosen(self, sample):
        if sample['prompt'] is not None and sample['chosen'] is not None:
            return " " + sample['prompt'] + " " + sample['chosen']
        return None

    def get_prompt_and_rejected(self, sample):
        if sample['prompt'] is not None and sample['rejected'] is not None:
            return " " + sample['prompt'] + " " + sample['rejected']
        return None


# Chinese dataset
class Wangrui6ZhihuKOLDataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "wangrui6/Zhihu-KOL"
        self.dataset_name_clean = "wangrui6_Zhihu_KOL"

    def get_train_data(self):
        from .data_utils import get_raw_dataset_split_index
        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(self.local_rank, self.output_path,
                                            self.dataset_name_clean,
                                            self.seed, "train_eval", "9,1", 0,
                                            len(dataset))
        dataset = Subset(dataset, index)
        return dataset

    def get_eval_data(self):
        from .data_utils import get_raw_dataset_split_index
        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(self.local_rank, self.output_path,
                                            self.dataset_name_clean,
                                            self.seed, "train_eval", "9,1", 1,
                                            len(dataset))
        dataset = Subset(dataset, index)
        return dataset

    def get_prompt(self, sample):
        if sample['INSTRUCTION'] is not None:
            return " Human: " + sample['INSTRUCTION'] + " Assistant:"
        return None

    def get_chosen(self, sample):
        if sample['RESPONSE'] is not None:
            return " " + sample['RESPONSE']
        return None

    def get_rejected(self, sample):
        print(
            f"Warning: dataset {self.dataset_name} does not include rejected response."
        )
        return None

    def get_prompt_and_chosen(self, sample):
        if sample['INSTRUCTION'] is not None and sample['RESPONSE'] is not None:
            return " Human: " + sample[
                'INSTRUCTION'] + " Assistant: " + sample['RESPONSE']
        return None

    def get_prompt_and_rejected(self, sample):
        print(
            f"Warning: dataset {self.dataset_name} does not include rejected response."
        )
        return None


# Chinese dataset
class CohereMiraclzhqueries2212Dataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "Cohere/miracl-zh-queries-22-12"
        self.dataset_name_clean = "Cohere_miracl_zh_queries_22_12"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["dev"]

    def get_prompt(self, sample):
        return " Human: " + sample['query'] + " Assistant:"

    def get_chosen(self, sample):
        return " " + sample['positive_passages'][0]['text']

    def get_rejected(self, sample):
        return " " + sample['negative_passages'][0]['text']

    def get_prompt_and_chosen(self, sample):
        return " Human: " + sample['query'] + " Assistant: " + sample[
            'positive_passages'][0]['text']

    def get_prompt_and_rejected(self, sample):
        return " Human: " + sample['query'] + " Assistant: " + sample[
            'negative_passages'][0]['text']


# Chinese dataset
class HelloSimpleAIHC3ChineseDataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "Hello-SimpleAI/HC3-Chinese"
        self.dataset_name_clean = "Hello_SimpleAI_HC3_Chinese"

    def get_train_data(self):
        from .data_utils import get_raw_dataset_split_index
        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(self.local_rank, self.output_path,
                                            self.dataset_name_clean,
                                            self.seed, "train_eval", "9,1", 0,
                                            len(dataset))
        dataset = Subset(dataset, index)
        return dataset

    def get_eval_data(self):
        from .data_utils import get_raw_dataset_split_index
        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(self.local_rank, self.output_path,
                                            self.dataset_name_clean,
                                            self.seed, "train_eval", "9,1", 1,
                                            len(dataset))
        dataset = Subset(dataset, index)
        return dataset

    def get_prompt(self, sample):
        if sample['question'] is not None:
            return " Human: " + sample['question'] + " Assistant:"
        return None

    def get_chosen(self, sample):
        if sample['human_answers'][0] is not None:
            return " " + sample['human_answers'][0]
        return None

    def get_rejected(self, sample):
        print(
            f"Warning: dataset {self.dataset_name} does not include rejected response."
        )
        return None

    def get_prompt_and_chosen(self, sample):
        if sample['question'] is not None and sample['human_answers'][
                0] is not None:
            return " Human: " + sample['question'] + " Assistant: " + sample[
                'human_answers'][0]
        return None

    def get_prompt_and_rejected(self, sample):
        print(
            f"Warning: dataset {self.dataset_name} does not include rejected response."
        )
        return None


# Chinese dataset
class MkqaChineseDataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "mkqa-Chinese"
        self.dataset_name_clean = "mkqa"

    def get_train_data(self):
        from .data_utils import get_raw_dataset_split_index
        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(self.local_rank, self.output_path,
                                            self.dataset_name_clean,
                                            self.seed, "train_eval", "9,1", 0,
                                            len(dataset))
        dataset = Subset(dataset, index)
        return dataset

    def get_eval_data(self):
        from .data_utils import get_raw_dataset_split_index
        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(self.local_rank, self.output_path,
                                            self.dataset_name_clean,
                                            self.seed, "train_eval", "9,1", 1,
                                            len(dataset))
        dataset = Subset(dataset, index)
        return dataset

    def get_prompt(self, sample):
        if sample['queries']['zh_cn'] is not None:
            return " Human: " + sample['queries']['zh_cn'] + " Assistant:"
        return None

    def get_chosen(self, sample):
        if sample['answers']['zh_cn'][0]['text'] is not None:
            return " " + sample['answers']['zh_cn'][0]['text']
        return None

    def get_rejected(self, sample):
        print(
            f"Warning: dataset {self.dataset_name} does not include rejected response."
        )
        return None

    def get_prompt_and_chosen(self, sample):
        if sample['queries']['zh_cn'] is not None and sample['answers'][
                'zh_cn'][0]['text'] is not None:
            return " Human: " + sample['queries'][
                'zh_cn'] + " Assistant: " + sample['answers']['zh_cn'][0][
                    'text']
        return None

    def get_prompt_and_rejected(self, sample):
        print(
            f"Warning: dataset {self.dataset_name} does not include rejected response."
        )
        return None


# Japanese dataset
class MkqaJapaneseDataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "mkqa-Japanese"
        self.dataset_name_clean = "mkqa"

    def get_train_data(self):
        from .data_utils import get_raw_dataset_split_index
        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(self.local_rank, self.output_path,
                                            self.dataset_name_clean,
                                            self.seed, "train_eval", "9,1", 0,
                                            len(dataset))
        dataset = Subset(dataset, index)
        return dataset

    def get_eval_data(self):
        from .data_utils import get_raw_dataset_split_index
        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(self.local_rank, self.output_path,
                                            self.dataset_name_clean,
                                            self.seed, "train_eval", "9,1", 1,
                                            len(dataset))
        dataset = Subset(dataset, index)
        return dataset

    def get_prompt(self, sample):
        if sample['queries']['ja'] is not None:
            return " Human: " + sample['queries']['ja'] + " Assistant:"
        return None

    def get_chosen(self, sample):
        if sample['answers']['ja'][0]['text'] is not None:
            return " " + sample['answers']['ja'][0]['text']
        return None

    def get_rejected(self, sample):
        print(
            f"Warning: dataset {self.dataset_name} does not include rejected response."
        )
        return None

    def get_prompt_and_chosen(self, sample):
        if sample['queries']['ja'] is not None and sample['answers']['ja'][0][
                'text'] is not None:
            return " Human: " + sample['queries'][
                'ja'] + " Assistant: " + sample['answers']['ja'][0]['text']
        return None

    def get_prompt_and_rejected(self, sample):
        print(
            f"Warning: dataset {self.dataset_name} does not include rejected response."
        )
        return None


# Japanese dataset
class CohereMiracljaqueries2212Dataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "Cohere/miracl-ja-queries-22-12"
        self.dataset_name_clean = "Cohere_miracl_ja_queries_22_12"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["dev"]

    def get_prompt(self, sample):
        return " Human: " + sample['query'] + " Assistant:"

    def get_chosen(self, sample):
        return " " + sample['positive_passages'][0]['text']

    def get_rejected(self, sample):
        return " " + sample['negative_passages'][0]['text']

    def get_prompt_and_chosen(self, sample):
        return " Human: " + sample['query'] + " Assistant: " + sample[
            'positive_passages'][0]['text']

    def get_prompt_and_rejected(self, sample):
        if len(sample['negative_passages']) > 0:
            return " Human: " + sample['query'] + " Assistant: " + sample[
                'negative_passages'][0]['text']
        return None


# Japanese dataset
class LmqgQgjaquadDataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "lmqg/qg_jaquad"
        self.dataset_name_clean = "lmqg_qg_jaquad"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["validation"]

    def get_prompt(self, sample):
        return " Human: " + sample['question'] + " Assistant:"

    def get_chosen(self, sample):
        return " " + sample['sentence']

    def get_rejected(self, sample):
        print(
            f"Warning: dataset {self.dataset_name} does not include rejected response."
        )
        return None

    def get_prompt_and_chosen(self, sample):
        return " Human: " + sample['question'] + " Assistant: " + sample[
            'sentence']

    def get_prompt_and_rejected(self, sample):
        print(
            f"Warning: dataset {self.dataset_name} does not include rejected response."
        )
        return None


# Japanese dataset
class LmqgQagjaquadDataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "lmqg/qag_jaquad"
        self.dataset_name_clean = "lmqg_qag_jaquad"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["validation"]

    def get_prompt(self, sample):
        return " Human: " + sample['questions'][0] + " Assistant:"

    def get_chosen(self, sample):
        return " " + sample['paragraph']

    def get_rejected(self, sample):
        print(
            f"Warning: dataset {self.dataset_name} does not include rejected response."
        )
        return None

    def get_prompt_and_chosen(self, sample):
        return " Human: " + sample['questions'][0] + " Assistant: " + sample[
            'paragraph']

    def get_prompt_and_rejected(self, sample):
        print(
            f"Warning: dataset {self.dataset_name} does not include rejected response."
        )
        return None


# English dataset
# Only for phase 3 training
class ToolDataset(PromptRawDataset):
    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = dataset_name
        self.dataset_name_clean = dataset_name
        self.tool_prompt = ""
        self.result_prompt = ""
        self.tool = True
        self.selfmade = True


class ToolMathDataset(ToolDataset):
    # TODO: dataset

    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "MathDataset"
        self.dataset_name_clean = "MathDataset"
        self.tool_prompt = """Your task is to create calls to a Calculator API to answer questions.
The calls should help you get information required to complete the text. 
If you do not think the API call is needed, you can just finish the Output with '[None()]'.
You can call the API by writing "[Calculator(expression)]" where "expression" is the expression to be computed. 
There are 6 examples below to show how to create Calculator API calls.
Your task is to finish the Output sentence in the [TASK] Input-Output pair.
example 1:
Input: what is 18 + 12 x 3?.
Output: [Calculator(18 + 12 * 3)].

example 2:
Input: The population is 658,893 people. What is the proportion of the national average of 5,763,868 people?
Output: [Calculator(658,893 / 5,763,868)]. 

example 3:
Input: Evaluate 1*(-4)/1*2450/(-700).
Output: [Calculator(1*(-4)/1*2450/(-700))]. 

example 4:
Input: A total of 252 qualifying matches were played, and 723 goals were scored. What is the average goals scored?
Output: [Calculator(723 / 252)].


example 5:
Input: I went to Paris in 1994 and stayed there until 2011. Which yead did I leaved Paris?
Output:[None()]. 

example 6:
Input: I went to Paris in 1994 and stayed there until 2011. How many years did I stay in Paris in total?
Output: [Calculator(2011 - 1994)].

[TASK]:
"""     
        self.result_prompt =   """
Your task is to answer questions with the help of a possible (there may be none) Calculator API.
The calls should help you get information required to complete the text. 
The API calls are written in the format of "[Calculator(expression)->result]" where "expression" is the expression to be computed and "result" is the computed result. 
If Calculator API is not needed, then the API call will be given as "[None()]", which means you do not need this information.
There are 6 examples below to show how to answer questions with the help of Calculator API calls. 
Your task is to finish the Output sentence in the [TASK] Input-Output pair.
Please start the Output sentence with "The answer is " and then put the answer after.

example 1:
Input: what is 18 + 12 x 3? [Calculator(18 + 12 * 3)->54].
Output: The answer is 54.

example 2:
Input: The population is 658,893 people. What is the proportion of the national average of 5,763,868 people? [Calculator(658,893 / 5,763,868)->0.114].
Output: The answer is 11.4%.

example 3:
Input: How many syllables are there in the word "apple"? [None()]. 
Output: There are two syllables in the word "apple". The answer is 2.

example 4:
Input: How many hours are there in 360 minutes? [Calculator(360 / 60)->6]
Output: The answer is 6.

example 5:
Input: I went to Paris in 1994 and stayed there until 2011. Which yead did I leaved Paris? [None()]. 
Output: You claimed to leave Paris in 2011. The answer is 2011.

example 6:
Input: A total of 252 qualifying matches were played, and 723 goals were scored. How many goals in average are scored in a match? [Calculator(723/252)->2.87]
Output: The answer is 2.87.

[TASK]:
"""

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_tool_prompt(self, sample):
        import re
        match = re.search(r"b'(.*?)\\n'", sample["question"])
        if match:
            text = match.group(1)
            return self.tool_prompt + "Input: " + text + "\nOutput:"
        else:
            return None

    def get_result_prompt(self):
        return self.result_prompt
    
    def get_chosen(self, sample):
        return NotImplementedError

    def get_rejected(self, sample):
        return NotImplementedError

    def get_prompt_and_chosen(self, sample):
        return NotImplementedError

    def get_prompt_and_rejected(self, sample):
        return NotImplementedError