# Federated Learning (FL) with Secure Multiparty Computation (SMPC)

## Use case: detect spam in messages while keeping them private

## Why Federated Learning?
Federated Learning protects data privacy. In FL, data is decentralized (not stored on one server) and does not leave its owners. A copy of a model is sent to each data owner. Model is trained there, without the model owner seeing data or uploading it to a central server.

## Why Secure Multiparty Computation?
Secure Multiparty Computation adds another layer of protection by keeping gradients private. Additive sharing generates secret shares of a value and distributes them among data owners. One data owner does not see the values of other owners. The parameters of the global model can still be updated.

## Example
Alice, Bob, and Jane receive lots of messages and would like to filter out spam. They want to keep their data private. We can help achieve these goals with FL and SMPC. Let us go step by step. 

### Import libraries
```python
import torch as th
import syft as sy
from torch import nn, optim
```
First, we need to import libraries.

### Create a hook to enable PyTorch functionalities
```python
hook = sy.TorchHook(th)
```
### Create workers (our data owners)
```python
bob = sy.VirtualWorker(hook, id="bob")
alice = sy.VirtualWorker(hook, id="alice")
jane = sy.VirtualWorker(hook, id="jane")
```
For our purposes, we use three workers. 

### Notify workers of each other's existence.
```python
bob.add_workers([alice,jane])
alice.add_workers([bob,jane])
jane.add_workers([alice,bob])
```
We are letting workers know of each other. This step is not needed for virtual workers, but it is mandatory for sockets.

### Dataset
```python
data = th.tensor([[1.,1,1],[0,1,1],[1,0,0],[0,0,0],[1,1,0],[0,1,0]], requires_grad=True)
target = th.tensor([[1.],[1],[0],[0],[1],[0]], requires_grad=True)
```
To keep things simple, we use a toy dataset. 

### Send data to workers
```python
bob_data = data[0:2].send(bob)
bob_target = target[0:2].send(bob)
alice_data = data[2:4].send(alice)
alice_target = target[2:4].send(alice)
jane_data = data[4:].send(jane)
jane_target = target[4:].send(jane)
```
This step is only for demonstration. Normally, workers have their own data. This is FL: data is decentralized, not uploaded to one server, and does not leave owners.

### Instantiate a model
```python
model = nn.Linear(3,1)
```
This is our global model consisting of three input neurons and one output. Three input neurons because our tensors have three features.

### Send a copy of the model to each worker
```python
for round_iter in range(5):
    bob_model = model.copy().send(bob)
    alice_model = model.copy().send(alice)
    jane_model = model.copy().send(jane)
```
We are sending a copy of the same global model to each worker. For the sake of simplicity, we decided to iterate only five times. This means we will go through the process of copying/sending models, training, and updating the global model five times.

### Define optimizer and learning rate for each worker
```python
    bob_opt = optim.SGD(params=bob_model.parameters(), lr=0.1)
    alice_opt = optim.SGD(params=alice_model.parameters(), lr=0.1)
    jane_opt = optim.SGD(params=jane_model.parameters(), lr=0.1)
```

### Train on each worker
```python
    for i in range(5):
        bob_opt.zero_grad() # zero out gradients.
        bob_pred = bob_model(bob_data) # generate predictions.
        bob_loss = ((bob_pred - bob_target)**2).sum() # calculate loss.
        bob_loss.backward() # backpropagate.
        bob_opt.step() # update weights.
        bob_loss = bob_loss.get().data # get the loss value.
```
This is the entire training for one worker. We zero out gradients to prevent them from accumulating continually. We then generate predictions and calculate loss. Our loss metric is here the mean squared error. We then backpropagate and update the weights for this worker in this iteration. We are also grabbing the loss value.

### Repeat for other workers   
```python
        alice_opt.zero_grad()
        alice_pred = alice_model(alice_data)
        alice_loss = ((alice_pred - alice_target)**2).sum()
        alice_loss.backward() 
        alice_opt.step()
        alice_loss = alice_loss.get().data
        
        jane_opt.zero_grad()
        jane_pred = jane_model(jane_data)
        jane_loss = ((jane_pred - jane_target)**2).sum()
        jane_loss.backward() 
        jane_opt.step()
        jane_loss = jane_loss.get().data
```

### Multiparty computation: additive sharing of weights
```python
    bw = bob_model.get().weight.data
    # encode floats as integers and share securely. Additve sharing requires integers.
    bw = bw.fix_prec().share(bob,alice,jane)
    bb = bob_model.bias.data
    bb = bb.fix_prec().share(bob,alice,jane)
```
We get weights and bias from one worker (Bob) and share them in an encrypted way. Share() generates secret shares of Bob's weight parameter and distributes them among workers. They cannot see directly Bob's parameter. Additive sharing requires integers. For this reason, we encode our weight and bias tensors with fix_prec() to enable interpreting floats as integers. 
    
### Repeat for other workers
```python
    aw = alice_model.get().weight.data
    aw = aw.fix_prec().share(bob,alice,jane)
    ab = alice_model.bias.data
    ab = ab.fix_prec().share(bob,alice,jane)

    # same thing for Jane.
    jw = jane_model.get().weight.data
    jw = jw.fix_prec().share(bob,alice,jane)
    jb = jane_model.bias.data
    jb = jb.fix_prec().share(bob,alice,jane)
```

### Update model parameters
```python
    with th.no_grad():
        print("avg weights", model.weight.set_(((bw + aw + jw) / 3).get().float_prec()))
        print("avg bias", model.bias.set_(((bb + ab + jb) / 3).get().float_prec()))
```
We average the gradients and update the parameters of the global model. We use float_prec() to decode the values that we previously encoded as integers with fix_prec().

We print the loss values.
```python
    print("Bob_loss:", str(bob_loss), "Alice_loss:", str(alice_loss), "Jane_loss:", str(jane_loss))
```
That's it.

## Conclusion
With our simple model, we keep data and gradients private and can train and update our model parameters to make it smarter with each iteration. This is a win-win for data owners and model owner. Data owners can enjoy good performance of spam detectors while knowing that their data is protected. Model owners know that it will keep improving, without them touching private data.

### Complete Example

You can find complete example [here](federated_learning_smpc.py).
