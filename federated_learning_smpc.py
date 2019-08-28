#!/usr/bin/env python 
"""
A simple example of federated learning with multi-party computation.
Federated learning is used to protect data privacy: data remains with its owners. 
Secure multiparty computation keeps the gradients private while allowing for
updating parameters of the global model.
"""

import torch as th
import syft as sy
from torch import nn, optim

# create a hook to enable PyTorch functionality.
hook = sy.TorchHook(th)

# create workers (data owners).
bob = sy.VirtualWorker(hook, id="bob")
alice = sy.VirtualWorker(hook, id="alice")
jane = sy.VirtualWorker(hook, id="jane")

# let workers know of each other. This is not needed for virtual workers, but it is mandatory for sockets.
bob.add_workers([alice,jane])
alice.add_workers([bob,jane])
jane.add_workers([alice,bob])

# toy dataset.
data = th.tensor([[1.,1,1],[0,1,1],[1,0,0],[0,0,0],[1,1,0],[0,1,0]], requires_grad=True)
target = th.tensor([[1.],[1],[0],[0],[1],[0]], requires_grad=True)

# send data to workers. This is for demo only: normally workers have their own data.
# This is now federated learning (data is decentralized and does not leave owners). 
bob_data = data[0:2].send(bob)
bob_target = target[0:2].send(bob)
alice_data = data[2:4].send(alice)
alice_target = target[2:4].send(alice)
jane_data = data[4:].send(jane)
jane_target = target[4:].send(jane)

# instantiate a global model.
model = nn.Linear(3,1)

# train, average gradients, and update parameters of the global model
# multiparty computation

for round_iter in range(5):
    
    # send a copy of the model to each worker.
    bob_model = model.copy().send(bob)
    alice_model = model.copy().send(alice)
    jane_model = model.copy().send(jane)

    # define optimizer and learning rate for each worker.
    bob_opt = optim.SGD(params=bob_model.parameters(), lr=0.1)
    alice_opt = optim.SGD(params=alice_model.parameters(), lr=0.1)
    jane_opt = optim.SGD(params=jane_model.parameters(), lr=0.1)

    # same operations for each worker.
    for i in range(5):
        bob_opt.zero_grad() # zero out gradients.
        bob_pred = bob_model(bob_data) # generate predictions.
        bob_loss = ((bob_pred - bob_target)**2).sum() # calculate loss.
        bob_loss.backward() # backpropagate.
        bob_opt.step() # update weights.
        bob_loss = bob_loss.get().data # get the loss value.
        

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

    # get weights and bias from Bob's model and share them in an encrypted way. 
    # These values are now invisible to other workers.
    bw = bob_model.get().weight.data
    # encode floats as integers and share securely. Additve sharing requires integers.
    bw = bw.fix_prec().share(bob,alice,jane)
    bb = bob_model.bias.data
    bb = bb.fix_prec().share(bob,alice,jane)
    
    # same thing for Alice.
    aw = alice_model.get().weight.data
    aw = aw.fix_prec().share(bob,alice,jane)
    ab = alice_model.bias.data
    ab = ab.fix_prec().share(bob,alice,jane)

    # same thing for Jane.
    jw = jane_model.get().weight.data
    jw = jw.fix_prec().share(bob,alice,jane)
    jb = jane_model.bias.data
    jb = jb.fix_prec().share(bob,alice,jane)
    
    # average and update the parameters of the global model.
    # use float_prec() to decode the values.
    with th.no_grad():
        print("avg weights", model.weight.set_(((bw + aw + jw) / 3).get().float_prec()))
        print("avg bias", model.bias.set_(((bb + ab + jb) / 3).get().float_prec()))

    #print("Bob_loss:", str(bob_loss), "Alice_loss:", str(alice_loss), "Jane_loss:", str(jane_loss))