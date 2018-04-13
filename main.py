import argparse
import os
from model import *
from data_loader import DataLoader
from evaluator import Evaluator
from trainer import Trainer


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Argument Parser for LAS')
    parser.add_argument("--data_dir", dest="data_dir", type=str, default="")

    parser.add_argument("--hidden_dimension", dest="hidden_dimension", type=int, default=512)
    parser.add_argument("--embedding_dimension", dest="embedding_dimension", type=int, default=512)
    parser.add_argument("--n_layers", dest="n_layers", type=int, default=1)

    parser.add_argument("--batch_size", dest="batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", dest="learning_rate", type=float, default=0.001)
    parser.add_argument("--num_epochs", dest="num_epochs", type=int, default=5)
    parser.add_argument("--clip_value", dest="clip_value", type=float, default=0)
    parser.add_argument("--learning_anneal", dest="learning_anneal", type=float, default=0.98)
    parser.add_argument("--wdecay", dest="wdecay", type=float, default=0.00001)
    parser.add_argument("--step_size", dest="step_size", type=int, default=30)
    parser.add_argument("--gamma", dest="gamma", type=int, default=10)
    parser.add_argument("--beam_width", dest="beam_width", type=int, default=32)
    parser.add_argument("--max_decoding_length", dest="max_decoding_length", type=int, default=300)
    parser.add_argument("--is_stochastic", dest="is_stochastic", type=int, default=1)

    parser.add_argument("--mode", dest="mode", type=int, default=0)
    model_dir_name = 'models'
    parser.add_argument("--model_dir", dest="model_dir", type=str, default=model_dir_name)
    parser.add_argument("--model_file_name", dest="model_file_name", type=str, default='bestModelWeights.t7')
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_dir_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    return parser.parse_args()


def main():
    params = parse_arguments()
    print("Constructing data loaders...")
    dl = DataLoader(params)
    evaluator = Evaluator(params, dl)
    print("Constructing data loaders...[OK]")

    if params.mode == 0:
        print("Training...")
        t = Trainer(params, dl, evaluator)
        t.train()
        print("Training...[OK]")
    elif params.mode == 1:
        print("Loading model...")
        model = LAS(params, len(dl.vocab), dl.max_seq_len)
        model_file_path = os.path.join(params.model_dir, params.model_file_name)
        model.load_state_dict(torch.load(model_file_path))
        if torch.cuda.is_available():
            model = model.cuda()
        print("Loading model...[OK]")

        print("Decoding test set...")
        evaluator.decode(model)
        print("Decoding on test set...[OK]")


if __name__ == '__main__':
    main()