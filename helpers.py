import matplotlib.pyplot as plt
import numpy as np
from numpy import ravel, reshape, swapaxes
import scipy.io
from sklearn import svm
from sklearn.metrics import confusion_matrix
from random import sample
from sklearn.metrics import confusion_matrix



def get_confusion_matrix(y_true, y_pred):
	print("Accuracy: ", (y_true == y_pred).sum() / len(y_true))
	return confusion_matrix(y_true, y_pred)

def plot_confusion_matrix(cm, title: str = None, save_name = None):
	fig = plt.figure()
	plt.matshow(cm)
	plt.title(title)
	plt.colorbar()
	plt.ylabel('True Label')
	plt.xlabel('Predicated Label')
	if save_name:
		plt.savefig('models/confusion_matrix'+str(save_name)+'.jpg')
	plt.show()


def plot_decision_boundary(pred_func,X,y, save_name=None):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx,yy=np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = np.array(pred_func(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)

    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    if y.shape[1] > 1:
        y = np.argmax(y,1)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.binary)
    
    plt.savefig('./decision_boundary'+str(save_name)+'.jpg')
    plt.show()
