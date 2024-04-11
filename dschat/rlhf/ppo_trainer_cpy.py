# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import torch
import torch.nn.functional as F
import time
import deepspeed
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from deepspeed.accelerator import get_accelerator

from dschat.utils.utils import print_rank_0
from transformers.generation import LogitsProcessor, LogitsProcessorList
from typing import Tuple, List, Union, Iterable
import numpy as np


def print_all_ranks(tag, value, rank):
    world_size = torch.distributed.get_world_size()
    all_tensor = torch.zeros(world_size, dtype=torch.float32).to(
        get_accelerator().current_device_name())
    all_tensor[rank] = value
    torch.distributed.all_reduce(all_tensor, op=torch.distributed.ReduceOp.SUM)
    print_rank_0(f'{tag} {all_tensor}', rank)


def get_model_norm(model):
    with torch.no_grad():
        total = 0.0
        for param in model.parameters():
            should_gather = hasattr(
                param,
                'ds_id') and param.ds_status == ZeroParamStatus.NOT_AVAILABLE
            with deepspeed.zero.GatheredParameters(param,
                                                   enabled=should_gather):
                total += float(param.float().norm())

    return total


def gather_log_probs(logits, labels):
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)


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

class DeepSpeedPPOTrainer():

    def __init__(self, rlhf_engine, args):
        self.rlhf_engine = rlhf_engine
        self.actor_model = self.rlhf_engine.actor
        self.critic_model = self.rlhf_engine.critic
        self.ref_model = self.rlhf_engine.ref
        self.reward_model = self.rlhf_engine.reward
        self.tokenizer = self.rlhf_engine.tokenizer
        self.args = args
        self.max_answer_seq_len = args.max_answer_seq_len
        self.end_of_conversation_token_id = self.tokenizer(
            args.end_of_conversation_token)['input_ids'][-1]
        self.z3_enabled = args.actor_zero_stage == 3
        self.compute_fp32_loss = self.args.compute_fp32_loss

        # In case the generated experience is not valid (too short), we use the last valid
        # generated experience. Alternatively, we can skip the step (on all workers).
        # For now, use the last valid experience which is a simpler solution
        self.last_generated_experience = None

        # Those value can be changed
        self.kl_ctl = getattr(args, "kl_ctl", 0.1)
        self.clip_reward_value = getattr(args, "clip_reward_value", 5)
        self.cliprange = 0.2
        self.cliprange_value = 0.2
        self.gamma = 1.0
        self.lam = 0.95
        self.generate_time = 0.0
        
        self.reward_scaling = getattr(args, "reward_scaling", False)
        self.norm_advantage = getattr(args, "norm_advantage", False)
        self.no_value_clip = getattr(args, "no_value_clip", False)
        self.kl_clip = getattr(args, "kl_clip", None)
        self.kl_appoximation = getattr(args, "kl_appoximation", False)
        device = torch.device(get_accelerator().device_name(), args.local_rank)
        self.device = device 
        if self.reward_scaling:
            self.adv_mean = torch.tensor(0.0, device=device, dtype=torch.float)
            self.adv_var = torch.tensor(0.0, device=device, dtype=torch.float)
            self.adv_cnt = torch.tensor(0.0, device=device, dtype=torch.float)
        self.stops = [
                self.tokenizer(".\n").input_ids,
                self.tokenizer(".\n\n").input_ids,
                self.tokenizer(".\n\n\n").input_ids,
                self.tokenizer(".\n\n\n\n").input_ids,
                self.tokenizer(".\n\n\n\n\n").input_ids,
                self.tokenizer(".\n\n\n\n\n\n").input_ids
            ]
        self.stop_logits_processor = StopWordsLogitsProcessor(self.stops, eos_token_id=self.tokenizer.eos_token_id)
        
        

    def _generate_sequence(self, prompts, mask, step):

        max_min_length = self.max_answer_seq_len + prompts.shape[1]

        # This has been added due to a probability/nan error that happens after
        # meta-llama/Llama-2-7b-hf enabled do_sample:
        # https://huggingface.co/meta-llama/Llama-2-7b-hf/commit/6fdf2e60f86ff2481f2241aaee459f85b5b0bbb9
        # if self.actor_model.module.config.model_type == "llama":
        #     kwargs = dict(do_sample=False)
        # else:
        #     kwargs = dict()
        kwargs = dict(do_sample=True)
        if self.args.actor_do_greedy:
            kwargs = dict(do_sample=False)

        with torch.no_grad():
            
            seq = self.actor_model.module.generate(
                prompts,
                attention_mask=mask,
                pad_token_id=self.tokenizer.pad_token_id,
                synced_gpus=self.z3_enabled,
                generation_config=None,
                logits_processor = LogitsProcessorList([self.stop_logits_processor]),
                output_scores=True,
                return_dict_in_generate=True,
                max_new_tokens=self.max_answer_seq_len,
                **kwargs)

        # Filter out seq with no answers (or very short). This happens when users directly use the pre-training ckpt without supervised finetuning
        # NOTE: this will causes each GPU has different number of examples
        seq, scores = seq["sequences"], seq["scores"]
        batch_size = seq.shape[0]
        prompt_length = prompts.shape[1]
        self.prompt_length = prompt_length
        ans = seq[:, prompt_length:]
        valid_ans_len = (ans != self.tokenizer.pad_token_id).sum(dim=-1)

        if self.args.print_answers and (step % self.args.print_answers_interval
                                        == 0):
            print(
                f"--- prompt --> step={step}, rank={torch.distributed.get_rank()}, {self.tokenizer.batch_decode(prompts, skip_special_tokens=True)}"
            )
            print(
                f"--- ans    --> step={step}, rank={torch.distributed.get_rank()}, {self.tokenizer.batch_decode(ans, skip_special_tokens=True)}"
            )

        out_seq = []
        for i in range(batch_size):
            if valid_ans_len[
                    i] <= 1:  # if the answer is shorter than 1 token, drop it
                print(
                    f'Dropping too short generated answer: {step=}: \n'
                    f'prompts: {self.tokenizer.batch_decode(prompts, skip_special_tokens=False)}\n'
                    f'answers: {self.tokenizer.batch_decode(ans, skip_special_tokens=False)}'
                )
                continue
            else:
                out_seq.append(seq[i:i + 1])

        if not out_seq:
            print(
                f'All generated results are too short for rank={self.args.local_rank} step={step}\n'
                f'-> prompts: {self.tokenizer.batch_decode(prompts, skip_special_tokens=False)}\n'
                f'-> answers: {self.tokenizer.batch_decode(ans, skip_special_tokens=False)}'
            )
            return None, None

        out_seq = torch.cat(out_seq, dim=0)  # concat output in the batch dim

        return out_seq, scores

    def generate_experience(self, prompts, mask, step, targets):
        self.eval()
        generate_start = time.time()
        seq, scores = self._generate_sequence(prompts, mask, step)
        generate_end = time.time()
        if seq is None:
            assert self.last_generated_experience is not None, f'Invalid generated experience at {step=}'
            prompts = self.last_generated_experience['prompts']
            seq = self.last_generated_experience['seq']
        else:
            self.last_generated_experience = {'prompts': prompts, 'seq': seq}
        self.train()

        pad_token_id = self.tokenizer.pad_token_id
        attention_mask = seq.not_equal(pad_token_id).long()
        
        # we need to mask sure of eos token
        # for i in range(attention_mask.shape[0]):
        #     mask_i = attention_mask[i].nonzero()
        #     if mask_i[-1].item() + 1 < attention_mask.shape[1]:
        #         attention_mask[i, mask_i[-1].item() + 1] = 1
        
        # if the last token is not eos, we replace it with eos
        # if the last token is eos, we keep it
        # if the last token is pad, we keep it 
        # seq[:, -1] = pad_token_id
        
        with torch.no_grad():
            output = self.actor_model(seq, attention_mask=attention_mask)
            output_ref = self.ref_model(seq, attention_mask=attention_mask)
            # reward_score = self.reward_model.forward_value(
            #     seq, attention_mask,
            #     prompt_length=self.prompt_length)['chosen_end_scores'].detach(
            #     )
            reward_score = self.reward_model(seq, scores, self.prompt_length, self.tokenizer, targets)
            values = self.critic_model.forward_value(
                seq, attention_mask, return_value_only=True).detach()[:, :-1]

        logits = output.logits
        logits_ref = output_ref.logits
        if self.compute_fp32_loss:
            logits = logits.to(torch.float)
            logits_ref = logits_ref.to(torch.float)

        self.generate_time = generate_end - generate_start

        return {
            'prompts': prompts,
            'logprobs': gather_log_probs(logits[:, :-1, :], seq[:, 1:]),
            'ref_logprobs': gather_log_probs(logits_ref[:, :-1, :], seq[:,
                                                                        1:]),
            'value': values,
            'rewards': reward_score,
            'input_ids': seq,
            "attention_mask": attention_mask
        }

    def compute_rewards(self, prompts, log_probs, ref_log_probs, reward_score,
                        action_mask):

        logr = ref_log_probs - log_probs
        if self.kl_clip and self.kl_clip > 0:
            logr = torch.clamp(logr, -self.kl_clip, self.kl_clip)
        if self.kl_appoximation:
            r = torch.exp(logr)
            kl_divergence_estimate = -self.kl_ctl * (r - 1 - logr)
        else:
            kl_divergence_estimate = -self.kl_ctl * (- logr)
        
        # kl_divergence_estimate = -self.kl_ctl * (- logr)
        rewards = kl_divergence_estimate
        start = prompts.shape[1] - 1
        ends = start + action_mask[:, start:].sum(1)
        
        if self.reward_scaling:
            reward_score_dtype = reward_score.dtype
            reward_score = reward_score.float()
            reward_list = [torch.zeros_like(reward_score, dtype=torch.float) for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(reward_list, reward_score)
            gathered_rewards = torch.cat(reward_list).view(-1)
            batch_count = torch.tensor(gathered_rewards.numel(), device=self.device, dtype=torch.float)
            batch_mean = torch.mean(gathered_rewards)
            batch_var = torch.var(gathered_rewards, correction=0)
            
            delta = batch_mean - self.adv_mean
            tot_count = self.adv_cnt + batch_count
            new_mean = self.adv_mean + delta * batch_count / tot_count
            m_a = self.adv_var * self.adv_cnt
            m_b = batch_var * batch_count
            M2 = m_a + m_b + delta**2 * self.adv_cnt * batch_count / tot_count
            new_var = M2 / tot_count
            self.adv_mean = new_mean
            self.adv_var = new_var
            self.adv_cnt = tot_count
            
            reward_score = (reward_score - self.adv_mean) / (torch.sqrt(self.adv_var) + torch.finfo(self.adv_var.dtype).eps)
            reward_score = reward_score.to(reward_score_dtype)
        
        reward_clip = torch.clamp(reward_score, -self.clip_reward_value,
                                  self.clip_reward_value)
        batch_size = log_probs.shape[0]
        for j in range(batch_size):
            rewards[j, start:ends[j]][-1] += reward_clip[j]

        return rewards

    def train_rlhf(self, inputs):
        # train the rlhf mode here
        ### process the old outputs
        prompts = inputs['prompts']
        log_probs = inputs['logprobs']
        ref_log_probs = inputs['ref_logprobs']
        reward_score = inputs['rewards']
        values = inputs['value']
        attention_mask = inputs['attention_mask']
        seq = inputs['input_ids']

        start = prompts.size()[-1] - 1
        action_mask = attention_mask[:, 1:]
        
        output_dict = dict()

        old_values = values
        with torch.no_grad():
            old_rewards = self.compute_rewards(prompts, log_probs,
                                               ref_log_probs, reward_score,
                                               action_mask)
            training_reward = old_rewards[:, start:][action_mask[:, start:].bool()].sum() / action_mask[:, start:].sum()
            output_dict["training_reward"] = training_reward
            
            gamma = torch.arange(seq.size(-1) - prompts.size(-1), device=self.device, dtype=old_rewards.dtype)
            gamma = torch.pow(self.gamma, gamma)
            sum_rewards = torch.sum(old_rewards[:, start:] * gamma * action_mask[:, start:], dim=-1).mean()
            output_dict["sum_rewards"] = sum_rewards
                
            
            ends = start + action_mask[:, start:].sum(1)
            # we need to zero out the reward and value after the end of the conversation
            # otherwise the advantage/return will be wrong
            for i in range(old_rewards.shape[0]):
                old_rewards[i, ends[i]:] = 0
                old_values[i, ends[i]:] = 0
            advantages, returns = self.get_advantages_and_returns(
                old_values, old_rewards, start)

        ### process the new outputs
        batch = {'input_ids': seq, "attention_mask": attention_mask}
        actor_prob = self.actor_model(**batch, use_cache=False).logits
        actor_log_prob = gather_log_probs(actor_prob[:, :-1, :], seq[:, 1:])
        actor_loss = self.actor_loss_fn(actor_log_prob[:, start:],
                                        log_probs[:, start:], advantages,
                                        action_mask[:, start:])
        self.actor_model.backward(actor_loss)

        if not self.args.align_overflow:
            self.actor_model.step()

        value = self.critic_model.forward_value(**batch,
                                                return_value_only=True,
                                                use_cache=False)[:, :-1]
        critic_loss = self.critic_loss_fn(value[:, start:], old_values[:,
                                                                       start:],
                                          returns, action_mask[:, start:])
        self.critic_model.backward(critic_loss)

        if self.args.align_overflow:
            actor_overflow = self.actor_model.optimizer.check_overflow(
                external=True)
            critic_overflow = self.critic_model.optimizer.check_overflow(
                external=True)

            rank = torch.distributed.get_rank()
            if actor_overflow and not critic_overflow:
                self.critic_model.optimizer.skip_step = True
                print_rank_0(
                    "OVERFLOW: actor overflow, skipping both actor and critic steps",
                    rank)
            elif not actor_overflow and critic_overflow:
                self.actor_model.optimizer.skip_step = True
                print_rank_0(
                    "OVERFLOW: critic overflow, skipping both actor and critic steps",
                    rank)
            elif actor_overflow and critic_overflow:
                print_rank_0(
                    "OVERFLOW: actor and critic overflow, skipping both actor and critic steps",
                    rank)
            self.actor_model.step()

        self.critic_model.step()
        
        with torch.no_grad():
            logr = ref_log_probs - log_probs
            logr = logr[:, start:]
            r = torch.exp(logr)
            kl_divergence_estimate = (r - 1 - logr)[action_mask[:, start:].bool()].sum() / action_mask[:, start:].sum()
            # kl_divergence_estimate = (- logr).mean()
            length = action_mask[:, start:].sum(1).float().mean()
            output_dict["kl_divergence_estimate"] = kl_divergence_estimate
            output_dict["length"] = length
            
            output_dict["returns_gae"] = returns[action_mask[:, start:].bool()].sum() / action_mask[:, start:].sum()
            output_dict["old_value"] = old_values[:, start:][action_mask[:, start:].bool()].sum() / action_mask[:, start:].sum()
            output_dict["value"] = value[:, start:][action_mask[:, start:].bool()].sum() / action_mask[:, start:].sum()

        return actor_loss, critic_loss, output_dict

    def get_overflow(self):
        # Overflow is not expected when using bf16
        # Therefore, DeepSpeed's BF16_Optimizer does not maintain an overflow indication
        if self.args.dtype == "bf16":
            return False, False

        actor_overflow = self.actor_model.optimizer.overflow
        critic_overflow = self.critic_model.optimizer.overflow

        return actor_overflow, critic_overflow

    def actor_loss_fn(self, logprobs, old_logprobs, advantages, mask):
        ## policy gradient loss
        log_ratio = (logprobs - old_logprobs) * mask
        ratio = torch.exp(log_ratio)
        
        if self.norm_advantage:
            adv = advantages[mask.bool()].view(-1)
            local_size = torch.tensor([adv.numel()], device=self.device, dtype=torch.long)
            size_list = [torch.tensor([0], device=self.device, dtype=torch.long) for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(size_list, local_size)
            
            max_size = max(size_list).item()
            padded = torch.cat([adv, torch.zeros(max_size - adv.numel(), device=self.device, dtype=adv.dtype)])
            gathered_tensors = [torch.zeros(max_size, device=self.device, dtype=adv.dtype) for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(gathered_tensors, padded)
            
            sliced_tensors = [t[:s.item()] for t, s in zip(gathered_tensors, size_list)]
            gathered_adv = torch.cat(sliced_tensors)
            
            advantages = (advantages - gathered_adv.mean()) / (gathered_adv.std() + torch.finfo(gathered_adv.dtype).eps)
            
        
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - self.cliprange,
                                             1.0 + self.cliprange)
        pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * mask) / mask.sum()
        return pg_loss

    def critic_loss_fn(self, values, old_values, returns, mask):
        ## value loss
        values_clipped = torch.clamp(
            values,
            old_values - self.cliprange_value,
            old_values + self.cliprange_value,
        )
        if self.compute_fp32_loss:
            values = values.float()
            values_clipped = values_clipped.float()
        vf_loss1 = (values - returns)**2
        vf_loss2 = (values_clipped - returns)**2
        vf_loss = 0.5 * torch.sum(
            torch.max(vf_loss1, vf_loss2) * mask) / mask.sum()
        return vf_loss

    def get_advantages_and_returns(self, values, rewards, start):
        # Adopted from https://github.com/CarperAI/trlx/blob/main/trlx/models/modeling_ppo.py#L134
        lastgaelam = 0
        advantages_reversed = []
        length = rewards.size()[-1]
        for t in reversed(range(start, length)):
            nextvalues = values[:, t + 1] if t < length - 1 else 0.0
            delta = rewards[:, t] + self.gamma * nextvalues - values[:, t]
            lastgaelam = delta + self.gamma * self.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values[:, start:]
        return advantages.detach(), returns

    def _validate_training_mode(self):
        assert self.actor_model.module.training
        assert self.critic_model.module.training

    def _validate_evaluation_mode(self):
        assert not self.actor_model.module.training
        assert not self.critic_model.module.training
        assert not self.ref_model.module.training
        assert not self.reward_model.module.training

    def train(self):
        self.actor_model.train()
        self.critic_model.train()

    def eval(self):
        self.actor_model.eval()
        self.critic_model.eval()
        # self.reward_model.eval()
        self.ref_model.eval()

    def dump_model_norms(self, tag):
        actor_model_norm = get_model_norm(self.actor_model)
        ref_model_norm = get_model_norm(self.ref_model)
        critic_model_norm = get_model_norm(self.critic_model)
        reward_model_norm = get_model_norm(self.reward_model)
        print_all_ranks(f'{tag} global_actor_model_norm', actor_model_norm,
                        self.args.local_rank)
        print_all_ranks(f'{tag} global_ref_model_norm', ref_model_norm,
                        self.args.local_rank)
        print_all_ranks(f'{tag} global_critic_model_norm', critic_model_norm,
                        self.args.local_rank)
        print_all_ranks(f'{tag} global_reward_model_norm', reward_model_norm,
                        self.args.local_rank)


class DeepSpeedPPOTrainerUnsupervised(DeepSpeedPPOTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_unsupervised(self, inputs, unsup_coef):
        # Train the unsupervised model here
        self._validate_training_mode()

        outputs = self.actor_model(**inputs, use_cache=False)
        loss = outputs.loss
        self.actor_model.backward(unsup_coef * loss)
        self.actor_model.step()

        return loss
