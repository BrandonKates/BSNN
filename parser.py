import argparse

class Parser():
	def __init__(self):
                self.parser = argparse.ArgumentParser(description='Train BSNN Model')
                self.parser.add_argument('--dataset', '-d', type=str, required=True, 
                                    help='which dataset do you want to train on?')

                self.parser.add_argument('--model', '-m', type=str, required=True, 
                                    help='which model do you want to run?')

                self.parser.add_argument('--epochs', type=int, default=100,
                                    help='number of epochs to train (default: 100)')

                self.parser.add_argument('--lr', type=float, default=0.01,
                                    help='learning rate (default: 0.01)')

                self.parser.add_argument('--momentum', type=float, default=0.5,
                                    help='SGD momentum')

                self.parser.add_argument('--no-cuda', action='store_true', default=False,
                                    help='disables CUDA training')

                self.parser.add_argument('--resize-input', action='store_true', 
                                        default=False)

                self.parser.add_argument('--no-orthogonal', action='store_true', default=False,
                                    help='initialise orthogonal weight')

                self.parser.add_argument('--seed', type=int, default=1,
                                    help='random seed')

                self.parser.add_argument('--log-interval', type=int, default=10,
                                    help='how many batches to wait before logging training status')

                self.parser.add_argument('--save-model', '-s', action='store_true', default=False,
                                    help='For Saving the current Model')

                self.parser.add_argument('--save-location', '-l', type=str, default='checkpoints/model.pt',
                                    help='Location to Save Model')

                self.parser.add_argument('--temp', '-t', 
                        help='temperature for softmax, required if using gumbel model')

                self.parser.add_argument('--deterministic', action='store_true', 
                        default=False, 
                        help='Run deterministic variant, if one exists')

                self.parser.add_argument('--batch-size', type=int, default=16,
		    help='input batch size for training')


	def parse(self):
		return self.parser.parse_args()

class DataArgs():
	def __init__(self, args):
		self.n = args.num_samples
		self.batch_size = args.batch_size
		self.test_batch_size = args.test_batch_size
		self.dataset = args.dataset


class ModelArgs():
        def __init__(self, args):
                self.no_cuda = args.no_cuda
                self.input_size
