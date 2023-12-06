import random
import wandb
import json
import random

from tqdm import tqdm
from opt.config import init_config
from opt.request import Request
from opt.reward import Reward
from opt.improve import Improve
from opt.select import Select


def generate_argmax_prompt(beam_candidate, val_data, reward_model, result_table):
    sample_data = val_data
    reward_list = [0] * len(beam_candidate)
    for index, prompt in enumerate(beam_candidate):
        reward = reward_model.calculate_reward(prompt, sample_data)
        reward_list[index] = reward
        result_table.add_data(prompt, reward)
    prompt_index = reward_list.index(max(reward_list))

    return beam_candidate[prompt_index]


if __name__ == '__main__':
    initial_prompt = "Based on the user's current session interactions, you need to answer the following subtasks step by step:\n" \
                     "1. Discover combinations of items within the session, where the number of combinations can be one or more.\n" \
                     "2. Based on the items within each combination, infer the user's interactive intent within each combination.\n" \
                     "3. Select the intent from the inferred ones that best represents the user's current preferences.\n" \
                     "4. Based on the selected intent, please rerank the 20 items in the candidate set according to the possibility of potential user interactions and show me your ranking results with item index.\n" \
                     "Note that the order of all items in the candidate set must be given, and the items for ranking must be within the candidate set.\n"
    
    inferring_reasons = "I'm trying to write a zero-shot recommender prompt.\n"\
                        "My current prompt is \"$prompt$\"\n"\
                        "But this prompt gets the following example wrong: $error_case$ "\
                        "give $num_feedbacks$ reasons why the prompt could have gotten this example wrong.\n"\
                        "Wrap each reason with <START> and <END>"

    refining_prompts = "I'm trying to write a zero-shot recommender prompt.\n"\
                       "My current prompt is \"$prompt$\"\n"\
                       "But this prompt gets the following example wrong: $error_case$\n"\
                       "Based on these example the problem with this prompt is that $reasons$.\n"\
                       "Based on the above information, please wrote one improved prompt.\n"\
                       "The prompt is wrapped with <START> and <END>.\n"\
                       "The new prompt is:"

    augumenting_prompts = "Generate a variation of the following instruction while keeping the semantic meaning.\n"\
                          "Input: $refined_prompt$\n"\
                          "Output:"

    conf = init_config()
    conf['initial_prompt'] = initial_prompt
    conf['inferring_reasons'] = inferring_reasons
    conf['refining_prompts'] = refining_prompts
    conf['augumenting_prompts'] = augumenting_prompts

    opt_request = Request(conf)
    if conf['use_wandb']:
        wandb.login(key=conf['wandb_api_key'])
        conf.pop('openai_api_key')
        run = wandb.init(
            project=f"PO4ISR_{conf['dataset']}_tune",
            config=conf,
        )
        text_table = wandb.Table(columns=["Input", "Prompt", "Reason", "Improved prompt", "Augumented prompt"])
        reward_table = wandb.Table(columns=["Prompt", "Reward"])
    else:
        text_table = None
    print("parameter initialization is complete")

    with open(f"../Dataset/{conf['dataset']}/Text/train_{conf['train_num']}.json", 'r') as json_file:
        train_data = json.load(json_file)
    with open(f"../Dataset/{conf['dataset']}/Text/valid.json", 'r') as json_file:
        val_data = json.load(json_file)

    beam_candidate = []
    prompt_candidate = []
    random.seed(conf['seed'])

    opt_reward = Reward(conf, opt_request)
    opt_improve = Improve(inferring_reasons, refining_prompts, augumenting_prompts, train_data, conf, opt_request)
    opt_select = Select(train_data, conf, opt_reward)

    print("==============")
    print("The apo algorithm is running...")
    print("==============")
    beam_candidate.append(initial_prompt)
    pbar = tqdm(range(conf['search_depth']))
    for i in pbar:
        pbar.set_description("search_depth " + str(i+1))
        prompt_candidate = []
        for prompt in beam_candidate:
            # Expand
            prompt_candidate = opt_improve.run(prompt, text_table)
            # Select
            beam_candidate = opt_select.run(prompt_candidate)
    pbar.close()
    # Argmax prompt
    new_prompt = generate_argmax_prompt(beam_candidate, val_data, opt_reward, reward_table)
    if conf['use_wandb']:
        text_table.add_data(" ", prompt, " ", " ", new_prompt)
        run.log({"texts": text_table})
        run.log({"rewards": reward_table})
    print("Optimize finished")
