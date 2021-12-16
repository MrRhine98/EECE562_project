import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat

class HMM(object):
	def __init__(self, nstate, nfeature, P, pi):
		self.curstate = -1
		self.nstate = nstate
		self.nfeature = nfeature
		self.mean = np.random.rand(nstate, nfeature)
		self.var = np.random.rand(nfeature)
		self.P = P
		self.pi = pi

	def _get_gaussian(self, y):
		nstate = self.nstate
		nfeature = self.nfeature
		sigma = np.diag(self.var)
		det_sigma = np.linalg.det(sigma)
		sigma_inv = np.diag(1/self.var)

		B = np.zeros((y.shape[1], nstate))
		for i in range(y.shape[1]):
			for j in range(nstate):
				bot = ((np.pi * 2)**nfeature * det_sigma)**0.5
				top = (y[:, i] - self.mean[j, :]).dot(sigma_inv).dot(y[:, i] - self.mean[j, :])
				B[i, j] = np.exp(-0.5 * top) / bot
			# normalize
			B[i, :] /= np.sum(B[i, :])
		return B


	def _forward(self, chroma):
		B = self._get_gaussian(chroma)
		alpha = np.zeros((chroma.shape[1], self.nstate))
		alpha[0, :] = B[0, :]
		for i in range(1, chroma.shape[1]):
			for j in range(self.nstate):
				alpha[i, j] = B[i, j] * alpha[i-1, :].dot(self.P[:, j])
			# normalize
			alpha[i, :] /= np.sum(alpha[i, :])
		# print(np.sum(alpha, axis=1))	
		return alpha

	def _backward(self, chroma):
		length = chroma.shape[1]
		B = self._get_gaussian(chroma)
		beta = np.zeros((chroma.shape[1], self.nstate))
		beta[-1, :] = np.ones(self.nstate)
		for k in range(length-1):
			i = length - k - 2
			for j in range(self.nstate):
				beta[i, j] = np.sum(beta[i+1, :] * self.P[j, :] * B[i+1, :])
			# normalize
			beta[i, :] /= np.sum(beta[i, :])	
		return beta	

	def _get_gamma(self, alpha, beta, chroma):
		length = chroma.shape[1]
		gamma1 = np.zeros((length, self.nstate))
		for t in range(length):
			gamma1[t, :] = (alpha[t, :] * beta[t, :]) / (alpha[t, :].dot(beta[t, :]))
		# print(np.sum(gamma1, axis=1))
			
		# gamma2
		B = self._get_gaussian(chroma)
		gamma2 = np.zeros((length-1, self.nstate, self.nstate))
		s = np.zeros(length-1)
		for t in range(length-1):
			for j in range(self.nstate):
				for k in range(self.nstate):
					gamma2[t, j, k] = alpha[t, j] * self.P[j, k] * B[t+1, k] * beta[t+1][k]
					s[t] += gamma2[t, j, k]
		for t in range(length-1):
			gamma2[t, :, :] /= s[t]
		return gamma1, gamma2		
	
		


	def train(self, filename):
		y, sr = librosa.load(filename, sr=22050)
		chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=4096, hop_length=2048)  # (12, len)
		fig, ax = plt.subplots()
		img = librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', ax=ax)
		fig.colorbar(img, ax=ax)
		ax.set(title='Chromagram')    

		
		alpha = self._forward(chroma)
		beta = self._backward(chroma)
		gamma1, gamma2 = self._get_gamma(alpha, beta, chroma)
		return gamma1, gamma2, chroma


	def update(self, gamma1, gamma2, chroma):
		_, length = np.shape(chroma)
		# update pi
		self.pi = gamma1[0, :]
		
		# update P
		P = np.sum(gamma1[:-1, :], axis=0)
		self.P = np.sum(gamma2, axis=0)
		for i in range(self.nstate):
			self.P[i, :] /= P[i]
				
		# update variance
		new_var = np.zeros(self.var.shape)
		for t in range(length):
			for i in range(self.nstate):
				new_var += gamma1[t, i] * (chroma[:, t] - self.mean[i, :])**2
		self.var = new_var / length

		# update mean
		new_mean = np.zeros(self.mean.shape)
		new_mean = chroma.dot(gamma1).T
		P = np.sum(gamma1, axis=0)
		for i in range(self.nstate):
			self.mean[i, :] = new_mean[i, :] / P[i] 
		return 

	def show_theta(self):
		print("******** pi ********")
		print(self.pi)
		print("******** P *********")
		print(self.P)
		print("******** var *******")
		print(self.var)
		print("******** mean ******")
		print(self.mean)


	def save_theta(self):
		theta_dic = {"pi": self.pi, "P": self.P, "var": self.var, "mean":self.mean}
		savemat("theta.mat", theta_dic)


def main():
	P = np.random.rand(6, 6)
	for i in range(6):
		P[i, :] /= sum(P[i, :])
	pi = np.array([0.65, 0.05, 0.05, 0.05, 0.05, 0.15])
	hmm = HMM(nstate=6, nfeature=12, P=P, pi=pi)
	for it in range(10):
		gamma1, gamma2, chroma = hmm.train("C_data.wav")
		hmm.update(gamma1, gamma2, chroma)
	hmm.show_theta()
	hmm.save_theta()	


if __name__ == "__main__":
	main()



# fig, ax = plt.subplots()
        # img = librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', ax=ax)
        # fig.colorbar(img, ax=ax)
        # ax.set(title='Chromagram')    
        # plt.show();

