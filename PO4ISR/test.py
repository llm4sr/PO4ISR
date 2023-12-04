import os
import wandb
from opt.eval import Eval
from opt.config import init_config
from opt.utils import load_eval_data


if __name__ == '__main__':
    test_prompt = "Based on the user's current session interactions, you need to answer the following subtasks step by step:\n" \
            "1. Discover combinations of items within the session, where the number of combinations can be one or more.\n" \
            "2. Based on the items within each combination, infer the user's interactive intent within each combination.\n" \
            "3. Select the intent from the inferred ones that best represents the user's current preferences.\n" \
            "4. Based on the selected intent, please rerank the 20 items in the candidate set according to the possibility of potential user interactions and show me your ranking results with item index.\n" \
            "Note that the order of all items in the candidate set must be given, and the items for ranking must be within the candidate set.\n"
    
    conf = init_config()
    test_data = load_eval_data(conf)

    key = conf['openai_api_key']
    if conf['use_wandb']:
        wandb.login(key=conf['wandb_api_key'])
        conf.pop('openai_api_key')
        run = wandb.init(
            project=f"PO4ISR_{conf['dataset']}_test",
            config=conf,
            name=f"seed_{conf['seed']}",
        )
        text_table = wandb.Table(columns=["Input", "Target", "Response"])
    else:
        text_table = None
    conf['openai_api_key'] = key

    eval_model = Eval(conf, test_data, text_table)
    results, target_rank_list, error_list = eval_model.run(test_prompt)

    result_save_path = f"./res/metric_res/{conf['dataset']}/"
    if not os.path.exists(result_save_path):
        os.makedirs(result_save_path)
    results.to_csv(f"{result_save_path}seed_{conf['seed']}.csv", index=False)
    
    if conf['use_wandb']:
        run.log({"texts": text_table})

