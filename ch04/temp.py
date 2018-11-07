import numpy as np

class Embedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        # self.grads = [np.random.randn(2,3)]
        self.idx = None
    
    def forward(self, idx):
        W, = self.params
        self.idx = idx
        out = W[idx]
        return out

    def backward(self, dout):
        dW, = self.grads
        print(dW)
        dW[...] = 0
        print(dW)
        dW[self.idx] = dout
        return None


W = np.random.randn(2,3)
# print(W)
embed = Embedding(W)
embed.backward(1)
