# Federated Learning (FL) with secure multiparty computation (SMPC)

## Problem/Use case: detect spam in messages while keeping them private and improving the model quality by updating its parameters

## Why Federated Learning? 
It protects data privacy because data does not leave its owners (workers). We send the model to data owners and train on their data, without seeing data or uploading it to a central server.

## Why SMPC? -- for another layer of protection
This is used to protect both data owners and the individual models and allows for improving the global model.
Secure additive sharing generates shares of a parameter and distributes them among data owners. In this way, the parameters of an individual model are protected: other data owners do not see my gradients.

SMPC requires integers, hence the need to use fix_precision, which enables intepreting floats as integers. 

Even encrypted, we still can update the overal parameters of the model. In this way, we keep data and gradients private and are able to continually update our global model. 

Data and gradients remain private, while our model gets smarter and smarter. Win-Win for data owners (they can enjoy good performance of spam detectors while knowing that their data is protected) and for model owners (their model keeps getting smarter and smarter with every iteration, without them touching private data).

## EXAMPLE

Lets say that we have data ....

### Import Libraries

First, we need to import libraries

```python
import torch as th
import syft as sy
from torch import nn, optim
```





### Complete Example

You can find complete example [here](federated_learning_smpc.py).
