import numpy as np

class Sequential:
    def __init__(self,layers:list):
        self.layers = layers
    def forward(self,x):
        out = []
        h=0
        for i in range(0,len(self.layers)):
            hl=self.layers[i].forward(h)
            h=hl[-1]
            out.extend(hl)
        return out

    def layer_count(self):
        ct=0
        for i in range(0,len(self.layers)):
            ct += self.layers[i].layer_count()
        return ct


class Split:
    def layer_count(self):
            cts = []
            for i in range(0,len(self.submodels)):
                cts.append(self.submodels[i].layer_count())
            
            assert len(list(set(cts)))==1, "Error in graph. Not all splits have equal length"
            return cts[0]
class SplitChoice(Split):
    def __init__(self,submodels:list = [], p:list=None):
        

        self.submodels=submodels
        if p is not None:
            if len(p) == len(submodels):
                self.p=p
            else:
                self.p=None
        else:
            self.p=None

    def forward(self,x):
        imod = np.random.choice(len(self.submodels),1,replace=True,p=self.p)[0]
        return self.submodels[imod].forward(x)

    
        
class SplitDecisionTree(Split):
    def __init__(self,submodels:list = [], fn = lambda x: 0):
        self.submodels=submodels
        self.fn=fn
    def forward(self,x):
        ii = self.fn(x)
        assert(ii >= 0 and ii <= len(self.submodels))
        return self.submodels[ii].forward(x)