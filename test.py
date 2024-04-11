TEST_PROMPT = """<|endoftext|><|endoftext|><|endoftext|><|endoftext|>Your task is to create calls to a Calculator API to answer questions.\nThe calls should help you get information required to complete the text. \nIf you do not think the API call is needed, you can just finish the Output with \'[None()]\'.\nYou can call the API by writing "[Calculator(expression)]" where "expression" is the expression to be computed. \nThere are 6 examples below to show how to create Calculator API calls.\nYour task is to finish the Output sentence in the [TASK] Input-Output pair.\nexample 1:\nInput: what is 18 + 12 x 3?.\nOutput: [Calculator(18 + 12 * 3)].\n\nexample 2:\nInput: The population is 658,893 people. What is the proportion of the national average of 5,763,868 people?\nOutput: [Calculator(658,893 / 5,763,868)]. \n\nexample 3:\nInput: Evaluate 1*(-4)/1*2450/(-700).\nOutput: [Calculator(1*(-4)/1*2450/(-700))]. \n\nexample 4:\nInput: A total of 252 qualifying matches were played, and 723 goals were scored. What is the average goals scored?\nOutput: [Calculator(723 / 252)].\n\n\nexample 5:\nInput: I went to Paris in 1994 and stayed there until 2011. Which yead did I leaved Paris?\nOutput:[None()]. \n\nexample 6:\nInput: I went to Paris in 1994 and stayed there until 2011. How many years did I stay in Paris in total?\nOutput: [Calculator(2011 - 1994)].\n\n[TASK]:\nInput: What is (-1)/(28/(1848/242))?\nOutput:"""
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
import torch
model = AutoModelForCausalLM.from_pretrained("/aiarena/gpfs/models/qwen2-1_8b", trust_remote_code=True, device_map="balanced")
tokenizer = AutoTokenizer.from_pretrained("/aiarena/gpfs/models/qwen2-1_8b", trust_remote_code=True)
prompt_ids = tokenizer(TEST_PROMPT, return_tensors="pt")
class MyStoppingCriteria(StoppingCriteria):
    def __init__(self, stop_token: int, encounters=3):
        super(MyStoppingCriteria, self).__init__()
        _stop_token = stop_token
        encounters = encounters

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs):
        stop_count = (_stop_token == input_ids[0]).sum().item()
        return stop_count >= encounters

my_stop = MyStoppingCriteria(stop_token=tokenizer.encode(".\n\n")[0], encounters=3)
stopping_criteria = StoppingCriteriaList([my_stop])
from transformers.generation import LogitsProcessor, LogitsProcessorList
from typing import Tuple, List, Union, Iterable
import numpy as np
class StopWordsLogitsProcessor(LogitsProcessor):
    """
    :class:`transformers.LogitsProcessor` that enforces that when specified sequences appear, stop geration.

    Args:
        stop_words_ids (:obj:`List[List[int]]`):
            List of list of token ids of stop ids. In order to get the tokens of the words
            that should not appear in the generated text, use :obj:`tokenizer(bad_word,
            add_prefix_space=True).input_ids`.
        eos_token_id (:obj:`int`):
            The id of the `end-of-sequence` token.
    """

    def __init__(self, stop_words_ids: Iterable[Iterable[int]], eos_token_id: int):

        if not isinstance(stop_words_ids, List) or len(stop_words_ids) == 0:
            raise ValueError(
                f"`stop_words_ids` has to be a non-emtpy list, but is {stop_words_ids}."
            )
        if any(not isinstance(bad_word_ids, list) for bad_word_ids in stop_words_ids):
            raise ValueError(
                f"`stop_words_ids` has to be a list of lists, but is {stop_words_ids}."
            )
        if any(
            any(
                (not isinstance(token_id, (int, np.integer)) or token_id < 0)
                for token_id in stop_word_ids
            )
            for stop_word_ids in stop_words_ids
        ):
            raise ValueError(
                f"Each list in `stop_words_ids` has to be a list of positive integers, but is {stop_words_ids}."
            )

        self.stop_words_ids = list(
            filter(
                lambda bad_token_seq: bad_token_seq != [eos_token_id], stop_words_ids
            )
        )
        self.eos_token_id = eos_token_id
        for stop_token_seq in self.stop_words_ids:
            assert (
                len(stop_token_seq) > 0
            ), "Stop words token sequences {} cannot have an empty list".format(
                stop_words_ids
            )

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        stopped_samples = self._calc_stopped_samples(input_ids)
        for i, should_stop in enumerate(stopped_samples):
            if should_stop:
                scores[i, self.eos_token_id] = float(2**15)
        return scores

    def _tokens_match(self, prev_tokens: torch.LongTensor, tokens: List[int]) -> bool:
        if len(tokens) == 0:
            # if bad word tokens is just one token always ban it
            return True
        elif len(tokens) > len(prev_tokens):
            # if bad word tokens are longer then prev input_ids they can't be equal
            return False
        elif prev_tokens[-len(tokens) :].tolist() == tokens:
            # if tokens match
            return True
        else:
            return False

    def _calc_stopped_samples(self, prev_input_ids: Iterable[int]) -> Iterable[int]:
        stopped_samples = []
        for prev_input_ids_slice in prev_input_ids:
            match = False
            for stop_token_seq in self.stop_words_ids:
                if self._tokens_match(prev_input_ids_slice, stop_token_seq):
                    # if tokens do not match continue
                    match = True
                    break
            stopped_samples.append(match)

        return stopped_samples


stops = [
                tokenizer(".\n").input_ids,
                tokenizer(".\n\n").input_ids,
                tokenizer(".\n\n\n").input_ids,
                tokenizer(".\n\n\n\n").input_ids,
                tokenizer(".\n\n\n\n\n").input_ids,
                tokenizer(".\n\n\n\n\n\n").input_ids
            ]
stop_logits_processor = StopWordsLogitsProcessor(stops, eos_token_id=tokenizer.eos_token_id)
return_ids = model.generate(**prompt_ids, pad_token_id=tokenizer.pad_token_id, max_length=1024, logits_processor = LogitsProcessorList([stop_logits_processor]),)
print(tokenizer.batch_decode(return_ids)[0])