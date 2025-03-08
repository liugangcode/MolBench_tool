import torch.nn as nn

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

class MLPPredictor:
    def __init__(
        self,
        in_features,
        out_features,
        hidden_features=300*4,
        act_layer=nn.GELU,
        bias=True,
        drop=0.5,
        lr=0.001,
        epochs=300,
        batch_size=128,
        device=None,
        task_type='classification'
    ):
        """
        Initialize the MLPPredictor with an MLP model.
        
        Args:
            in_features: Input dimension
            hidden_features: Hidden layer dimension (defaults to in_features)
            out_features: Output dimension (defaults to in_features)
            act_layer: Activation function
            bias: Whether to use bias in linear layers
            drop: Dropout rate
            lr: Learning rate
            epochs: Number of training epochs
            batch_size: Batch size for training
            device: Device to run the model on ('cuda' or 'cpu')
            task_type: Type of task ('classification' or 'regression')
        """
        if device is None:
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.model = MLP(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            act_layer=act_layer,
            bias=bias,
            drop=drop
        )
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.model.to(device)
        self.criterion = nn.BCEWithLogitsLoss() if task_type == 'classification' else nn.L1Loss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.task_type = task_type

    def fit(self, X, y, verbose=False):
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        X = torch.FloatTensor(X).to(self.device)
        y = torch.FloatTensor(y).to(self.device)
        
        # Create dataset and dataloader
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        # Training loop
        self.model.train()
        if verbose:
            epoch_iterator = tqdm(range(self.epochs), desc="Training")
        else:
            epoch_iterator = range(self.epochs)
        
        for epoch in epoch_iterator:
            running_loss = 0.0
            for inputs, targets in dataloader:
                self.optimizer.zero_grad()
                is_valid = ~torch.isnan(targets)
                outputs = self.model(inputs)
                loss = self.criterion(outputs[is_valid], targets[is_valid])
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            
            if verbose:
                # Update progress bar with loss information
                epoch_iterator.set_postfix({"loss": f"{running_loss/len(dataloader):.4f}"})
            
        return self
    
    def predict_proba(self, X):
        assert isinstance(X, np.ndarray)
        X = torch.FloatTensor(X).to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X)            
            if self.task_type == 'classification':
                probs = torch.sigmoid(outputs)
                return probs.cpu().numpy()
            else:
                return outputs.cpu().numpy()
    
    def predict(self, X):
        probs = self.predict_proba(X)
        if self.task_type == 'classification':
            return (probs > 0.5).astype(int)
        else:
            return probs

class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        bias=True,
        drop=0.5,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        linear_layer = nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias)
        self.bn1 = nn.BatchNorm1d(hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        return x