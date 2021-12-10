import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


class HMM(object):
	def __init__(self, nstate, nfeature, P, pi):
		self.curstate = -1
		self.nstate = nstate
		self.nfeature = nfeature
		self.mean = np.random.rand(nstate, nfeature)
		self.var = np.random.rand(nstate, nfeature)
		self.P = P
		self.pi = pi

	def _get_gaussian(self, y, j):
		sigma = np.diag(self.var[j, :])
		sigma_inv = np.diag(1/self.var[j, :])
		c = (2*np.pi)**(-self.nfeature/2) * (np.prod(self.var[j, :])**-0.5)
		e = np.exp(-0.5*(y - self.mean[j, :]).dot(sigma_inv).dot(y - self.mean[j, :]))
		# print(c*e)
		return c*e

	def _forward(self, chroma):
		_, length = np.shape(chroma)
		norm = np.zeros(length)
		alpha = list()
		last_alpha = self.pi
		for i in range(length):
			observation = chroma[:, i]
			new_alpha = np.zeros(last_alpha.shape)
			for j in range(self.nstate):
				gaussian = self._get_gaussian(observation, j) 
				new_alpha[j] = last_alpha.dot(self.P[:, j]) * gaussian
			# normalization
			norm[i] = sum(new_alpha)
			new_alpha /= norm[i]
			alpha.append(new_alpha)
			last_alpha = new_alpha
		return norm, alpha

	def _backward(self, chroma, norm):
		_, length = np.shape(chroma)
		beta = list()
		last_beta = np.ones(self.nstate)
		beta.append(last_beta)
		for l in range(length-1):		
			t = length - l - 1
			observation = chroma[:, t]
			for i in range(self.nstate):
				new_beta = np.zeros(last_beta.shape)
				b = np.zeros(self.nstate)
				for g in range(self.nstate):
					b[g] = self._get_gaussian(observation, g)
				new_beta[i] = sum(last_beta * self.P[i, :] * b)
			new_beta /= norm[t-1]
			# print(new_beta)
			beta.append(new_beta)
			last_beta = new_beta
		beta.reverse()
		return beta	

	def _get_gamma(self, alpha, beta, chroma):
		length = len(alpha)
		gamma1 = np.zeros((length, self.nstate))
		for t in range(length):
			gamma1[t, :] = (alpha[t] * beta[t]) / (alpha[t].dot(beta[t]))
		
		# gamma2
		gamma2 = np.zeros((length-1, self.nstate, self.nstate))
		for t in range(length-1):
			for j in range(self.nstate):
				for k in range(self.nstate):
					gamma2[t, j, k] = alpha[t][j] * self.P[j, k] * self._get_gaussian(chroma[:, t+1], k) * beta[t+1][k]
		for j in range(self.nstate):
			for k in range(self.nstate):
				gamma2[:, j, k] /= np.sum(np.sum(gamma2, axis=2), axis=1)

		return gamma1, gamma2		
	
		


	def train(self, filename):
		y, sr = librosa.load(filename, sr=11025)
		chroma = librosa.feature.chroma_stft(y=y, sr=sr)  # (12, len)
		fig, ax = plt.subplots()
		img = librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', ax=ax)
		fig.colorbar(img, ax=ax)
		ax.set(title='Chromagram')    
		plt.show();



		norm, alpha = self._forward(chroma)
		beta = self._backward(chroma, norm)
		gamma1, gamma2 = self._get_gamma(alpha, beta, chroma)
		return gamma1, gamma2, chroma


	def update(self, gamma1, gamma2, chroma):
		_, length = np.shape(chroma)
		# update P
		P = np.sum(gamma1, axis=0)
		self.P = np.sum(gamma2, axis=0)
		for i in range(self.nstate):
			self.P[i, :] /= P[i]
				
		# update variance
		new_var = np.zeros(self.var.shape)
		for t in range(length):
			for i in range(self.nstate):
				new_var += np.outer(gamma1[i], (chroma[:, t] - self.mean[i])**2)
		self.var = new_var / length

		# update mean
		new_mean = np.zeros(self.mean.shape)
		for t in range(length):
			new_mean += np.outer(gamma1[t], chroma[:, t])
		for i in range(self.nstate):
			new_mean[i, :] / P[i]
		self.mean = new_mean
	

def main():
	P = np.random.rand(6, 6)
	for i in range(6):
		P[i, :] /= sum(P[i, :])
	pi = np.array([0.5, 0.05, 0.05, 0.05, 0.05, 0.3])
	hmm = HMM(nstate=6, nfeature=12, P=P, pi=pi)
	gamma1, gamma2, chroma = hmm.train("C_piano.wav")
	hmm.update(gamma1, gamma2, chroma)

if __name__ == "__main__":
	main()



# fig, ax = plt.subplots()
        # img = librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', ax=ax)
        # fig.colorbar(img, ax=ax)
        # ax.set(title='Chromagram')    
        # plt.show();

