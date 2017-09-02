# Activation functions for the CTHERESA package

import numpy as np
# import matplotlib.pyplot as plt
# import time

class Activations():
	def sigmoid(self, x, deriv = False):
		if deriv:
			return self.sigmoid(x)*(1-self.sigmoid(x))
		return 1/(1+np.exp(-x))

	def tanh(self, x, deriv = False):
		if deriv:
			return 1 - self.tanh(x)**2
		return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

	def relu(self, x, deriv = False):
		if deriv:
			return np.sign(self.relu(x))
		return np.maximum(x,0)

	def leaky_relu(self, x, alpha, deriv = False):
		if deriv:
			return np.maximum(np.sign(x), np.sign(x)*alpha)
		return np.maximum(x,x*alpha)

	def linear(self, x, deriv = False):
		if deriv:
			return 1
		return x

if __name__ == "__main__":
	activations = Activations()
	activations.leaky_relu(np.array([-3,9]),0.05)
