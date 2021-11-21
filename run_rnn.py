import os
import argparse
from strategies.rnn import RNN

def str2bool(v):
    if isinstance(v,bool):
        return v
    elif v == 'True':
        return True
    elif v == 'False':
        return False
    else:
        print("[ERROR] should be True or False, but got {}".format(v))
        exit()


def run(args):
    print("======== arguments ==========")
    print(args)
    print("=============================")
    
    model = RNN(name=args.name, stock_id=args.stock_id,
                input_strategy=args.input_strategy, training_model=args.training_model,
                stateful=True, batch_size=args.batch_size, input_size=args.input_size,
                return_sequences=args.return_sequences, rnn_output_size=args.rnn_output_size,
                rnn_activation=args.rnn_activation,
                dense_output_size=args.dense_output_size, dense_activation=args.dense_activation,
                loss=args.loss,
                sample_path=args.sample_path, patience=args.patience, reload=args.reload
                )
    model.fit(epochs=args.epochs)
    
    model.predict("test", args.save)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=False, default="rnn", type=str)
    parser.add_argument('--stock_id', required=False, default="000002.SZ", type=str)
    parser.add_argument('--input_strategy', required=False, default=1, type=int)
    parser.add_argument('--training_model', required=False, default="LSTM", type=str)
    # parser.add_argument('--stateful', required=False, default=True, type=str2bool)
    parser.add_argument('--return_sequences', required=False, default=True, type=str2bool)
    parser.add_argument('--batch_size', required=False, default=1, type=int)
    parser.add_argument('--input_size', required=False, default=33, type=int)
    parser.add_argument('--rnn_output_size', required=False, default=[16, 8, 4], type=int, nargs='+',)
    parser.add_argument('--rnn_activation', required=False, default="sigmoid", type=str)
    parser.add_argument('--dense_output_size', required=False, default=[], type=int, nargs='+',)
    parser.add_argument('--dense_activation', required=False, default="", type=str)
    parser.add_argument('--loss', required=False, default="rmse", type=str)
    parser.add_argument('--sample_path', required=False, default="/root/sampleTest", type=str)
    parser.add_argument('--patience', required=False, default=5, type=int)
    parser.add_argument('--reload', required=False, default=True, type=str2bool)
    parser.add_argument('--epochs', required=False, default=100, type=int)
    parser.add_argument('--save', required=False, default=True, type=str2bool)
    
    args = parser.parse_args()
    
    if args.stock_id.lower() == "all":
        stocks_list = os.listdir(os.path.join(args.sample_path, "train"))
        for stock_id in stocks_list:
            args.stock_id = stock_id[:-4]
            run(args)
    else:
        run(args)

