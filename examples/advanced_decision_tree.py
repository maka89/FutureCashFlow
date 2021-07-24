from futurecashflow import *
  

# Year 1: Cash Flow between (0 and 1) 
layer1 = UniformLayer(0,1)

# Scenario 1 for years 2-4:
l1 = Sequential([ UniformLayer(1,2),UniformLayer(1,2),UniformLayer(1,2)])

#Scenario 2 for years 2-4:
l2 = Sequential([ UniformLayer(-0.1,0.5),UniformLayer(-0.1,0.5),UniformLayer(-0.1,0.5)])


#Select scenario 1 / 2 based on if cash flow for year 1 is smaller or larger than 0.5.

def fn(x):
    if x <= 0.5:
        return 0
    else:
        return 1

layers = [layer1, SplitDecisionTree([l1,l2],fn)]


# Create model. Optionally add a discount_rate.
model = Model(layers,discount_rate=0.02)

print(model.n_layers)

# Year0 cash flow = 1.0 (No effect)
# Run 500000 MonteCarlo simulations.
# Return cumulative cash flow. out has shape (n_sims, n_layers)=(500000,4).
out = model.run(1.0,500000,cumulative=True)

#Plot histogram of total discounted cash flow.
import matplotlib.pyplot as plt
plt.hist(out[:,-1],bins=200,density=True)
plt.show()