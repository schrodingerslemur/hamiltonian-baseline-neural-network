Files and their usage:

data/data_prep : Creates pendulum time, px, py, qx, qy, H dataset based on derived physics equations
data/dataset : Defines HamiltonianDataset class used to load data into neural network
data/all csv files : CSV files created from data_prep

network/neural_net : Neural network trained to output H from inputs of px, py, qx, and qy

trainer and tester : As name suggests. Trainer creates model.pth file and tester loads it.
maintraintest : Calls train and test functions

finite_calculations/finite : Calculates partial derivative of H with respect to px, py, qx, and qy given coordinates

Rooms for improvement:
- Use mean for tolerance in tester.py
- Compare to actual data
- Move operations to GPU
