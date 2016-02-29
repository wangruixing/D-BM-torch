require 'dbm'
require 'torch'

function render_output(i,k)
  energy.insert(energy,getn(energy),dbm_test.total_energy())
  print(i+"cycle, layer "+k+ " cycle energy: " +energy[-1])
end

function render_supervised(i)
  entropy.insert(entropy,getn(entropy),dbm_test.total_entropy())
  accuracy.insert(accuracy,getn(accuracy),1-torch.mean(torch.abs(torch.round(dbm_test.predict_probs(dataset))-labels)))
  print(i+" cycle entropy: "+ entropy[-1]+" cycle accuracy: "+accuracy[-1])
end

dataset = torch.round(torch.rand(10000, 1))
labels = 1- dataset
dataset = torch.cat(dataset,1-dataset,1)
dataset = torch.cat(dataset,torch.ones(getn(dataset),1),1)
print("dataset shape :" + dataset.size) ---- dataset.shape?

energy = {}
entropy = {}
accuracy = {}

print("initializing model")

dbm_test =dbm ()

for i = 1,3,1 do
  for k = 0,10,1 do
    print("beginning boltzmann training of model")
    dbm_test.train_unsupervised(k)
    render_output(i,k)
  end
end
dbm_test.learning_rate = 1.0
dbm_test.addlayer(1)
dbm_test.labels = labels

render_output(-1,4)
render_supervised(-1)
for i = 0,20,1 do     -----train backprop
  layers = 1
  dbm_test.train_backprop(layers)
  render_output(i,4)
  render_supervised(i)
end
for i = 0,20,1 do
  dbm_test.train_backprop()
  render_output(i,4)
  render_supervised(i)
end
