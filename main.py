import argparse
import torch
import math
import time
from core.model import RUNModel
from core.utils import load_pickle


parser = argparse.ArgumentParser(description='PyTorch Attend Image Captions Model')

parser.add_argument('--data', type=str, default='./data',
                    help='location of the data corpus')
parser.add_argument('--save', type=str,  default='./models/parameters.pkl',
                    help='path to save the final model')
parser.add_argument('--val_samples', type=str,  default='./models',
                    help='path to save the random choice val samples')
parser.add_argument('--test_samples', type=str,  default='./models/test_samples.pkl',
                    help='path to save the all test samples')
parser.add_argument('--loss_log', type=str,  default='./models/loss.pkl',
                    help='note the best val loss')

parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='batch size')
parser.add_argument('--batch_norm', type=int, default=196, metavar='N',
                    help='batch_norm size')
parser.add_argument('--epochs', type=int, default=500,
                    help='upper epoch limit')
parser.add_argument('--embedding_size', type=int, default=512,
                    help='size of word embeddings')
parser.add_argument('--time_step', type=int, default=16, metavar='N',
                    help='the LSTM time_step')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--hidden_dim', type=int, default=1024,
                    help='hidden_dim')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')

parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')

parser.add_argument('--demo_feat', type=float, default=1e-3,
                    help='demo feature')

args = parser.parse_args()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

# best_val_loss = load_pickle(args.loss_log)
# model = RUNModel(args,best_val_loss)
# run the model
def main():
    best_val_loss = load_pickle(args.loss_log)
    model = RUNModel(args, best_val_loss)
    try:
        for epoch in range(args.epochs):
            epoch_start_time = time.time()

            # start training
            model.train(epoch)

            # start val and compute bleu score
            args.lr,val_loss,val_captions = model.val(epoch)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                  'valid ppl {:8.2f}  | learning rate {:10.8f} |'.format(epoch, (time.time() - epoch_start_time),
                                             val_loss, math.exp(val_loss), args.lr))
            print('-' * 89)
        model.test()
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')


if __name__ == "__main__":
    main()