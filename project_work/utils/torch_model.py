import torch


class MLPtorch(torch.nn.Module):
    def __init__(self):
        super(MLPtorch, self).__init__()
        self.fc1 = torch.nn.Linear(2, 16)
        self.fc2 = torch.nn.Linear(16, 2)

    def forward(self, x):
        res = torch.sigmoid(self.fc1(x))
        res = self.fc2(res)
        return res


class MLPNet(torch.nn.Module):
    def __init__(self, n_feature, n_hidden1, n_hidden2, n_hidden3, n_output):
        super(MLPNet, self).__init__()
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden1)
        self.hidden2 = torch.nn.Linear(n_hidden1, n_hidden2)
        self.hidden3 = torch.nn.Linear(n_hidden2, n_hidden3)
        self.predict = torch.nn.Linear(n_hidden3, n_output)
        self.lrelu   = torch.nn.LeakyReLU()
        self.tanh    = torch.nn.Tanh()

    def forward(self, x):
        x = self.lrelu(self.hidden1(x))
        x = self.tanh(self.hidden2(x))
        x = self.lrelu(self.hidden3(x))
        x = self.predict(x)
        return x

#class Net(torch.nn.Module):
#    def __init__(self, n_feature, n_hidden1, n_hidden2, n_output):
#        super(Net, self).__init__()
#        self.hidden1 = torch.nn.Linear(n_feature, n_hidden1)
#        self.hidden2 = torch.nn.Linear(n_hidden1, n_hidden2)
#        self.predict = torch.nn.Linear(n_hidden2, n_output)

#    def forward(self, x):
#        x = torch.sigmoid(self.hidden1(x))
#        x = torch.sigmoid(self.hidden2(x))
#        x = self.predict(x)
#        return x
