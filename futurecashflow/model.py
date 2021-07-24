import numpy as np
class Model:
    def __init__(self,layers:list,discount_rate = None):
        self.layers=layers
        self.discount_rate=discount_rate
        self.n_layers=self.layer_count()
    def forward(self,x):
        h=x
        out=[]
        for i in range(0,len(self.layers)):

            hl=self.layers[i].forward(h)
            h=hl[-1]
            out.extend(hl)

        return out
    def layer_count(self):
        ct=0
        for i in range(0,len(self.layers)):
            ct+=self.layers[i].layer_count()
        return ct
    def run(self,x,n_sims:int=10000,cumulative:bool=True):

        vals = []
        for i in range(0,n_sims):
            vals.append(self.forward(x))
        out=np.array(vals)
        if self.discount_rate:
            darr = 1/(1+self.discount_rate)
            darr = darr**(np.arange(out.shape[1])+1)
            out = out*darr.reshape(1,-1)
        if cumulative:
            out = np.cumsum(out,1)
        return out