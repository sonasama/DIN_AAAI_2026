import argparse


def parameter_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="0", help="gpu number or cpu")
    parser.add_argument("--path", type=str, default="./datasets/", help="Dataset path")
    parser.add_argument("--dataset", type=str, default="MNIST", help="Dataset name")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed")
    parser.add_argument("--fix_seed", action='store_true', default=True, help="xx")
    parser.add_argument("--save", action='store_true', default=False, help="xx")
    parser.add_argument("--normal", action='store_true', default=True, help="xx")
    parser.add_argument("--plot", action='store_true', default=False, help="xx")
    parser.add_argument("--type", type=str, default="feat", help="fus type")
    parser.add_argument("--model_type", type=str, default="3", help="fus type")

    parser.add_argument("--n_repeated", type=int, default=1, help="Repeated times")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-05, help="Weight decay")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout")
    parser.add_argument("--num_layers", type=int, default=2, help="num of layer")

    parser.add_argument("--knns", type=int, default=10, help="knn")   #UCI set 1
    parser.add_argument("--ratio", type=float, default=0.1, help="Label ratio")
    parser.add_argument("--num_epoch", type=int, default=500, help="Training epochs")

    parser.add_argument("--r_view", type=int, default=2, help="reduction view")
    parser.add_argument("--r_feat", type=int, default=2, help="reduction feature")

    parser.add_argument("--hidden_dims", type=int, default=128, help="hidden dimensions")

    parser.add_argument("--noisy_edge", action='store_true', default=False, help="xx")
    parser.add_argument("--noisy_fea", action='store_true', default=False, help="xx")
    parser.add_argument("--noise_rate", type=float, default=0.1, help="xx")

    args = parser.parse_args()

    return args
