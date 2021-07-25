import numpy as np

class CoreLayer:
    def __init__(self,additive:bool=False,relative:bool=False):
        self.additive=additive

    def forward(self,x):
        self.out=self.fwd(x)

        if self.additive:
            if self.relative:
                return [x*(1+self.out)]
            else:
                return [x+self.out]
        else:
            return [self.out]

    def layer_count(self):
        return 1


class NormalLayer(CoreLayer):
    def __init__(self,mu:float=0.0,std:float=1.0,additive:bool=False):
        super().__init__(additive)
        self.mu=mu
        self.std=std
    
    def fwd(self,x):
        return np.random.normal(self.mu,self.std)

    
class LogNormalLayer(CoreLayer):
    def __init__(self,mu:float=0.0,std:float=1.0,additive:bool=False):
        super().__init__(additive)
        self.mu = mu
        self.std=std
    def fwd(self,x):
        return np.random.lognormal(self.mu,self.std)
    

class ExponentialLayer(CoreLayer):
    def __init__(self,scale:float=1.0,additive:bool=False):
        super().__init__(additive)
        self.scale=scale
    def forward(self,x):
        return np.random.exponential(self.scale)

class UniformLayer(CoreLayer):
    def __init__(self,minval:float=0.0,maxval:float=1.0,additive:bool=False):
        super().__init__(additive)
        self.minval=minval
        self.maxval=maxval

    def fwd(self,x):
        return np.random.rand()*(self.maxval-self.minval) + self.minval

class ChoiceLayer(CoreLayer):
    def __init__(self,values:list = [0,1], p:list=None,additive:bool=False):
        super().__init__(additive)
        self.values = values
        if p is not None:
            if len(p) == len(values):
                self.p=p
            else:
                self.p=None
        else:
            self.p=None

    def fwd(self,x):
        return np.random.choice(self.values,1,replace=True,p=self.p)[0]