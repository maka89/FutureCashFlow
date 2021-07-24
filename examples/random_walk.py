from futurecashflow import *
  
# Every year cash flow is expected to increase by 0.1.
# Change in cash flow year over year is assumed to be normally distributed. 
layers = [NormalLayer(0.1,0.2,additive=True),NormalLayer(0.1,0.2,additive=True),NormalLayer(0.1,0.2,additive=True)]

# Create model. Optionally add a discount_rate.
model = Model(layers,discount_rate=0.02)

# Year0 cash flow = 1.0 (Acts as an offset in this particular scenario)
# Run 500000 MonteCarlo simulations.
# Return cumulative cash flow. out has shape (n_sims, n_periods)=(500000,3).
out = model.run(1.0,500000,cumulative=True)

#Plot histogram of total discounted cash flow.
import matplotlib.pyplot as plt
plt.hist(out[:,-1],bins=200,density=True)
plt.show()