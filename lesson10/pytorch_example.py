import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T

class LinearClassifier(nn.Module):
    def __init__(self, lr, n_classes, input_dims):
        super(LinearClassifier, self).__init__()

        self.fc1 = nn.Linear(*input_dims, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, n_classes)

        # Author: self.parameters() from inherited class Module
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.CrossEntropyLoss() #nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        # Author: pytorch have different tensors for cuda/cpu devices
        self.to(self.device)

    def forward(self, data):
        layer1 = F.sigmoid(self.fc1(data))
        layer2 = F.sigmoid(self.fc2(layer1))
        # Author: CrossEntropy will take care of activation for us...
        layer3 = self.fc3(layer2)

        return layer3

    # Author: DeepQ will get different parameters (state, action, ...)  
    def learn(self, data, labels):
        # Author: pytorch keep values from previous iteration but they are
        #   not required/needed - so we clean it up.
        self.optimizer.zero_grad()

        # Author: pytorch requires conversion of types prior to use external
        #   types adapting them to required target tensor types.

        # pytorch have Tensor class as well which defaults to float and/or 64bit
        # internal types. We use tensor to save memory using regular 32bit
        # whenever possible to save memory.
        data = T.tensor(data).to(self.device)
        labels = T.tensor(labels).to(self.device)

        # Author: get predictions and evaluate cost (how far predictions are
        #   from actual labels
        predictions = self.forward(data)
        cost = self.loss(predictions, labels)

        # Author: backpropagate cost and add a step on our optimizer.
        # These two calls are critical for learn loop.
        cost.backward()
        self.optimizer.step()







