import torch
from torch.utils.data import DataLoader
from dataset import MusicDataset
from model import UNetMask

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = MusicDataset("data/mixtures",
                       "data/source1",
                       "data/source2")

loader = DataLoader(dataset, batch_size=1, shuffle=True)

model = UNetMask().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

for epoch in range(20):
    total_loss = 0

    for X, M in loader:
        X = X.to(device)
        M = M.to(device)

        optimizer.zero_grad()
        M_pred = model(X)
        loss = loss_fn(M_pred, M)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch}, Loss {total_loss:.4f}")

torch.save(model.state_dict(), "models/mask_model.pt")