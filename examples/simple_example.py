from futurecashflow import *
  
# Year1: CashFlow between 0 and 1, 
# Year2: CashFlow between 0.1 and 1.2
# Year3: CashFlow between 0.2 and 1.4
layers = [UniformLayer(0,1),UniformLayer(0.1,1.2),UniformLayer(0.2,1.4)]

# Create model. Optionally add a discount_rate.
model = Model(layers,discount_rate=0.02)

# Year0 cash flow = 1.0 (Has no effect in this particular scenario)
# Run 500000 MonteCarlo simulations.
# Return cumulative cash flow. out has shape (n_sims, n_periods)=(500000,3).
out = model.run(1.0,500000,cumulative=True)

#Plot histogram of total discounted cash flow.
import matplotlib.pyplot as plt
plt.hist(out[:,-1],bins=200,density=True)
plt.show()