import numpy as np, torch as th, collections, random
from torch.autograd import Variable

class ExperienceReplay(object):
    def __init__(self, capacity=2**20):
        self.capacity = capacity
        self.buffer = collections.deque(maxlen=self.capacity)
        self.ordering = ['states','actions','rewards','sprimes','terminals']
    def retain(self, state, action, reward, sprime, terminal):
        s = np.asarray(state).astype(np.float32)
        a = np.asarray(action).astype(np.float32)
        r = np.asarray(reward).astype(np.float32)
        s_= np.asarray(sprime).astype(np.float32)
        t = np.asarray(terminal).astype(np.float32)
        self.buffer.append((s,a,r,s_,t))
    def wipe(self):
        self.buffer = collections.deque(maxlen=self.capacity)
    def minibatch(self, bsize=1, device=th.device('cpu')):
        S,A,R,S_,T = self.sample(bsize)
        S = Variable(th.from_numpy(S).float()).to(device)
        A = Variable(th.from_numpy(A).float()).to(device)
        R = Variable(th.from_numpy(R).float()).to(device)
        S_= Variable(th.from_numpy(S_).float()).to(device)
        T = Variable(th.from_numpy(T).float()).to(device)
        return S,A,R,S_,T
    def recent(self, bsize=1, device=th.device('cpu')):
        s,a,r,s_,t = np.asarray(self.buffer)[-bsize,:]
        S = Variable(th.from_numpy(np.asarray(s)).float()).to(device)
        A = Variable(th.from_numpy(np.asarray(a)).float()).view(1).to(device)
        R = Variable(th.from_numpy(np.asarray(r)).float()).view(1).to(device)
        S_= Variable(th.from_numpy(np.asarray(s_)).float()).to(device)
        T = Variable(th.from_numpy(np.asarray(t)).float()).to(device)
        return S,A,R,S_,T
    def sequence(self, axis='rewards'):
        return np.asarray(self.buffer)[:,self.ordering.index(axis)]
    def sample(self, bsize=1):
        s,a,r,s_,t = zip(*random.sample(self.buffer, bsize))
        return np.array(s),np.array(a),\
               np.array(r),np.array(s_),np.array(t)
    def __len__(self):
        return len(self.buffer)
