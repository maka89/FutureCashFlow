from futurecashflow import *


l1_1 = Sequential([NormalLayer(1,0.2),NormalLayer(0.9,0.25),NormalLayer(0.8,0.3),NormalLayer(0.7,0.4)])
l1_2 = Sequential([NormalLayer(2,0.2),NormalLayer(2.1,0.3),NormalLayer(2.2,0.4),NormalLayer(2.4,0.6)])

l2 = SplitChoice([l1_1,l1_2],[0.6,0.4])

layers = [l2]

model = Model(layers,discount_rate=0.02)

out = model.run(1.0,500000,cumulative=True)


import matplotlib.pyplot as plt
mu = np.mean(out[:,-1])

plt.hist(out[:,-1],bins=200,density=True)
plt.vlines([mu],0.0,0.4,'r')
plt.xlabel("Discounted Cash Flow")
plt.ylabel("Probability")
plt.legend(["Mean","Density"])
plt.title("Discounted Cash Flow")
plt.show()