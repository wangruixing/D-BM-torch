require 'torch'
--- dataset: binary valued data matrix
--- layers :a list with the size of each hidden layer
--- labels :the outputs for each data row
--- the number of markov chains 
--- learning_rate:start learning rate
function create(dataset, labels = torch.tensor({}), batch_size = 50, layers = [10,2],fantasy_count = 100, learning_rate = 0.0001)
  self.dataset = dataset
  self.labels = labels
  self.datapts = dataset:size(1)
  self.batch_size = batch_size
  self.features = dataset:size(2)
  self.learning_rate = learning_rate
  self.layers = {}
  self.layers.insert(layers, getn(layers),{
                                           'size': self.features,
                                           'fantasy': torch.rand(fantasy_count,self.features), 
                                           'mu':0,
                                           'bias':self.sigma_inverse(torch.mean(dataset)).reshape(1,self.features),
                    })  
  
end

---stochastic annealing scheduler
function next_learning_rate(rate)
  return 1.0/(1.0/rate+1)
end
function randint(mini,maxi,size)
  rand_arr = {}
  for i = 1,size,1 do
    rand_arr[i] = torch.rand(1)*(maxi-mini)+mini
  end
  return rand_arr
end

function l2_pressure(weights)
  norms = torch.cmax(torch.cmin(torch.sqrt(torch.sum(weights*weights)),10000),0.00001)
  norms = torch.floor(1/norms.reshape(norms:size(1),1))
  norms = norms.T.repeatTensor(weights:size(1))
  out = norms * weights
  return  -0.01 * out   
end
--- sigmod function  logistic regression
function sigma(x)
  x = torch.cmax(torch.cmin(x, 100),-100)
  return 1/(1+torch.exp(-x))
end
function sigma_inverse(x)
  x = torch.cmax(torch.cmin(x, 0.99999),0.00001)
  return torch.log(x/(1-x))
end

--- quick bootstrapper
function data_sample(num)
  if(self.labels.shape[0]>0) then
    return (self.dataset[randint(0,self.dataset.size(1),num)], self.labels[randint(0,dataset.size(1),num)],{})  
  else
    return (self.dataset[randint(0,self.dataset.size(1),num)],{}) 
  end

end
--- returns activations for probabilities
function sample(fn , args)
  temp = fn(args)
  temp_cutoff = torch.rand(temp.size)
  return(temp >temp_cutoff)
end

----- the propagates through the net
function predict_probs(test, prop_uncertainty = False, omit_layers = 0)
  out = test
  for i = 1,getn(self.layers)-omit_layers 1 do
    W = self.layers[i]['W']
    bias = self.layers[i]['bias']
    out = self._predict(W,bias,out)
    if (not prop_uncertainty and i< getn(self.layers)-1) then  out = torch.round(out)
    end
  end
  return out             
end

function _predict(W,bias,inputs)
  return self.sigma(bias+ torch.dot(inputs,W))
end
--- the energy of a given layer with a given input and output vector
function _energy(v,W,h,bv,vh)
  return torch.mean(-torch.dot(v,bv.T)- torch.dot(h,bh.T))- torch.dot(torch.dot(v,W),h)
end


--- Whole DBM's energy with given inputs and outpus
function internal_energy(v,hs)
  temp = self._energy(v, self.layers[1]['W'],hs[1],self.layers[1]['bias'],self.layers[2]['bias'])
  for i = 2, getn(self.layers)-1, 1 do
    temp = temp + self._energy(hs[i-1],self.layers[i+1]['W'],hs[i],self.layers[i]['bias'],self.layers[i+1]['bias'])
  end
  return temp
end
--- the network's energy with the inpur activiation
function energy(v)
  hs  = [torch.round(self.sigma(self.layers[1]['bias']+torch.dot(v,self.layers[1]['W'])))]
  for i = 3, getn(self.layers), 1 do
    torch.cat(torch.round(self.sigma(self.layers[1]['bias']+torch.dot(hs[-1],self.layers[i]['W']))))
  end
  return self.internal_energy(v,hs)  --?
end

---return total energy of the stored dataset
function total_energy()
  return self.energy(self.dataset)
end

---return the total entropy of the dataset 
function total_entropy()
  pred = torch.cmax(torch.cmin(self.predict_probs(self.dataset),0.9999),0.0001)
  return torch.sum(self.self.labels*torch.log(pred) + (1-self.labels)*torch.log(1-pred))
end


function prob_given_vis(W, vs, bias, double = false)
  if(double)  then return self.sigma(2*(bias +torch.dot(vs, W)))
  else  return self.sigma(bias +torch.dot(vs, W))
  end
end

function prob_given_out(W, hs, bias, double = false)
  if(double) then  return self.sigma(2*(bias+torch.dot(hs,W.T)))
  else   return self.sigma(bias +torch.dot(hs, W.T))
  end
end

---gibbs sampler updates
function gibbs_update(gibbs_iterations = 10, layers = nil)
  if layers == nil then layers = getn(self.layers)
  end
  for j= 0, gibbs_iteration,1
    for i= 1, layers, 1 do
      double = i%2
      active = self.layers[i-1]['fantasy']
      bias = self.layers[i]['W']
      W = self.layers[i]['W']
      self.layers[i]['fantasy'] = self.sample(self.prob_given_vis,(W,active,bias,double))
    end
    for i= layers-1, 1, -1 do
      double ~= i%2 
      active = self.layers[i]['fantasy']
      bias = self.layers[i-1]['bias']
      W = self.layers[i]['W']
      self.layers[i-1]['fantasy'] = self.sample(self.prob_given_out,(W,active,bias,double))          
    end
  
end
--- BP algorithm
function train_backprop(train_iteration = 100, weight = 1, layers = nil)
  for iter= 0,train_iteraions,1 do
    rate = self.learning_rate
    rows, labels = self.data_sample(self.batch_size)
    self.backprop_step(self.dataset, self.labels, rate*weight, train = layers)
  end
end
--- gradient descent
function backprop_step(data, labels,rate, momentum_decay = 0, train_layers = nil)
  min = 0
  layers = getn(self.layers)
  if train_layers ~= nil then min = layers-train_layers - 1
  end
  -- Backpropagate
  for layer= layers-1,min,-1 do
    act = self.predict_probs(data)
    use_W = self.layers[layers-1]['W']
    weight_errors = torch.mean((act-labels)*act*(1-act))*use_W
    bias_errors = (act-labels)*act*(1-act)
  end
  --Actuals
    for iter= layers-1, layer, -1 do
      use_W = self.layers[iter-1]['W']
      prior_act = self.predict_probs(data, omit_layers = layers-iter)
      prior_act = torch.mean(prior_act*(1-prior_act))
      prior_act = prior_act.reshape(prior_act:size(0),1)   ---- shape?
      weight_errors = prior_act* weight_errors
      weight_errors = torch.dot(use_W,weight_errors)
    W = self.layers[layer]['W']
    b = self.layers[layer]['bias']
    gradient = 1.0/data
    W = W - rate *gradient
    self.layers[layer]['W']= W + self.l2_pressure(W)
  end  
end
---Training
function train_unsupervised(layer,train_iterations = 10000, gibbs_iterations = 10)
  layers = getn(self.layers)
  if(layer>=layers) then print("Not enough layers to train specified layer"+tostring(layers)+" vs " + tostring(layer)) end
  for iter= 0,train_iteration,1 do
    self.gibbs_update(gibbs_iterations,layer)
    data, labels = self.data_sample(self.batch_size)
    rate = self.learning_rate
    self.learning = self.next_learning_rate(self.learning_rate)
    previous = torch.round(self.predict_probs(data, omit_layers = layers-layer))
    bias = self.layers[layer]['bias']
    mu = bias+torch.dot(previous, self.layers[layer]['W'])
    bias_part = torch.mean(mu).reshape(bias.size)
    self.layers[layer]['bias'] = bias + rate*(bias_part-bias)
    if layer%2 == 0 then mu = self.sigma(2*mu)
    else mu = self.sigma(mu)
    end
    self.layers[layer]['mu'] = mu
    gradient_part = -1.0/self.fantasy_count * torch.dot(self.layers[layer-1]['fantasy'].T,self.layers[layer]['fantasy'])

    W = (self.layers[layer]['W'] + rate *gradient_part + rate *approx_part)
    self.layers[layer]['W'] = W + self.l2_pressure(W)
  end
end
function add_layer(size)
  hidden = {[size] = "size", [mu] = 0}
  above = self.layers[-1][size]
  hidden['W'] = torch.randn(above,size)
  hidden['bias'] = torch.randn(1,size)
  hidden['momentum'] = torch.zeros(above,size)
  hidden['fantasy'] = torch.rand(self.fantasy_count,size)       -----astype
  self.layers.insert(hidden, getn(self.layers))
end

