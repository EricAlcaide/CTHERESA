# Initializations functions for the CTHERESA package

import numpy as np
# import matplotlib.pyplot as plt
# import time

class Initializations():
	def zeros(self, shape):
		return np.zeros((shape))

	def ones(self, shape):
		return np.ones((shape))

	def random(self, shape):
		return np.random.randn(shape[0],shape[1])

	def xavier(self, shape):
		return np.random.randn(shape[0],shape[1])*np.sqrt(1/shape[1])

	def xavier_relu(self, shape):
		return np.random.randn(shape[0],shape[1])*np.sqrt(2/shape[1])

	def xavier_variant(self, shape):
		return np.random.randn(shape[0],shape[1])*np.sqrt(2/np.sum(shape))


if __name__ == "__main__":
	initializations = Initializations()
	np.random.seed(1)
	print(initializations.xavier_variant((4,5)))
	np.random.seed(1)
	print(initializations.xavier_relu((4,5)))
	np.random.seed(1)
	print(initializations.xavier((4,5)))
