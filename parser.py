import argparse

class Parser():
	def __init__(self):
		self.parser = argparse.ArgumentParser(description='train a stochastic model')
		self.parser.add_argument('--dataset', '-d', type=str, required=True, 
		                    help='which dataset do you want to train on?')
		self.parser.add_argument('--batch-size', type=int, default=16, metavar='N',
		                        help='input batch size for training (default: 64)')
		self.parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
		                    help='input batch size for testing (default: 1000)')
		self.parser.add_argument('--epochs', type=int, default=100, metavar='N',
		                    help='number of epochs to train (default: 10)')
		self.parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
		                    help='learning rate (default: 0.01)')
		self.parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
		                    help='SGD momentum (default: 0.5)')
		self.parser.add_argument('--no-cuda', action='store_true', default=False,
		                    help='disables CUDA training')
		self.parser.add_argument('--seed', type=int, default=1, metavar='S',
		                    help='random seed (default: 1)')
		self.parser.add_argument('--log-interval', type=int, default=10, metavar='N',
		                    help='how many batches to wait before logging training status')

		self.parser.add_argument('--save-model', '-s', action='store_true', default=False,
		                    help='For Saving the current Model')
		self.parser.add_argument('--save-location', '-l', action='store_true', default='checkpoints/model.pt',
		                    help='Location to Save Model')

	def parse(self):
		return self.parser.parse_args()

''' Ignore this for now '''
class Args():
    def __init__(self, n, input_size, hidden_size, num_classes, num_epochs, batch_size, learning_rate):
        self.n = n
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate