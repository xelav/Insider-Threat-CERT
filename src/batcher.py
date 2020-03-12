import numpy as np

class CertBatcher:
	def __init__(self, actions, malicious, seed=0, batch_size=16):

		self.actions = actions
		self.malicious = malicious

		self.num_items = actions.shape[0]

		self.indices = np.arange(self.num_items)
		self.batch_size = batch_size
		self.rnd = np.random.RandomState(seed)
		self.rnd.shuffle(self.indices)
		self.ptr = 0
		self.bi = 0  # batch index

	def __iter__(self):
		# self._reset()
		return self
	
	def __len__(self):
		return self.num_items // self.batch_size
	
	def _reset(self):
		self.rnd.shuffle(self.indices)
		self.ptr = 0
		self.bi = 0
		
	def __next__(self):
		if self.ptr + self.batch_size > self.num_items:
			# self._reset()
			raise StopIteration()

		if self.bi >= 10:
			raise StopIteration()
		current_indecies = self.indices[self.ptr:self.ptr+self.batch_size]
		result = (self.actions[current_indecies], self.malicious[current_indecies])
		self.ptr += self.batch_size
		self.bi += 1
		return result