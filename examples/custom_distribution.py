from futurecashflow import *
  

# This example uses a custom probability distribution.


values = [2,2,1,1,1,3,3,4]

layer1 = ChoiceLayer(values)
layer2 = ChoiceLayer(values)

# This also works:
values_alt = [1,2,3,4]
p = np.array([3,2,2,1]) #Count of each occurence in values_alt
p = p / np.sum(p) # Normalize to get probabilities

layer3 = ChoiceLayer(values_alt,p=p)



layers = [layer1,layer2,layer3]
# Create model. Optionally add a discount_rate.
model = Model(layers,discount_rate=0.02)

# Year0 cash flow = 1.0 (Has no effect in this particular scenario)
# Run 200000 MonteCarlo simulations.
# Return cumulative cash flow. out has shape (n_sims, n_periods)=(200000,3).
out = model.run(1.0,200000,cumulative=True)

#Plot histogram of total discounted cash flow.
import matplotlib.pyplot as plt
plt.hist(out[:,-1],bins=200,density=True)
plt.show()