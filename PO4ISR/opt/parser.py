import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='APO4Rec')
    parser.add_argument('--reward_func', 
                        type=str,
                        default='ndcg',
                        help='how the reward is calculated, options: cos_similarity/ndcg')
    parser.add_argument('--model', 
                        type=str,
                        default='gpt-3.5-turbo',
                        help='which model as recommender, options: gpt-3.5-turbo')
    parser.add_argument('--seed', 
                        type=int,
                        default=42,
                        help='options: 42, 625, 2023, 0, 10')
    parser.add_argument('--candidate_size', 
                        type=int,
                        default=20,
                        help='options: 10, 20')
    parser.add_argument('--dataset', 
                        type=str,
                        default='bundle',
                        help='use which datset: bundle/games/ml-1m')
    parser.add_argument('--train_num', 
                        type=int,
                        default=50,
                        help='options: 50,100,150')
    parser.add_argument('--batch_size', 
                        type=int,
                        default=16,
                        help='options: 16,32')

    args = parser.parse_args()
    
    return args