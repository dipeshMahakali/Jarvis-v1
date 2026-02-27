import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from snn.encoding import poisson_encode
from snn.lif import LIFLayer


class SNNClassifier(nn.Module):
    def __init__(self, in_features: int, hidden: int, out_features: int, timesteps: int):
        super().__init__()
        self.timesteps = timesteps

        self.fc1 = nn.Linear(in_features, hidden)
        self.lif1 = LIFLayer(size=hidden, tau=0.95, threshold=1.0, reset=0.0, refractory_steps=0)

        self.fc2 = nn.Linear(hidden, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]

        x_spk = poisson_encode(x, timesteps=self.timesteps)

        s1 = self.lif1.init_state(batch_size=batch, device=x.device, dtype=x.dtype)

        out_sum = 0.0
        for t in range(self.timesteps):
            h = self.fc1(x_spk[t])
            spk1, s1 = self.lif1(h, s1)
            out_sum = out_sum + self.fc2(spk1)

        return out_sum / float(self.timesteps)


def make_toy_dataset(n: int = 4096, in_features: int = 32, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, in_features, generator=g)

    w = torch.linspace(-1.0, 1.0, in_features)
    y = (x @ w > 0.0).to(torch.long)

    x = (x - x.min()) / (x.max() - x.min() + 1e-6)
    return x, y


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=-1)
    return (preds == y).float().mean().item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--timesteps", type=int, default=25)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    x, y = make_toy_dataset()
    n_train = int(0.8 * x.shape[0])

    ds_train = TensorDataset(x[:n_train], y[:n_train])
    ds_test = TensorDataset(x[n_train:], y[n_train:])

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True)
    dl_test = DataLoader(ds_test, batch_size=args.batch_size)

    model = SNNClassifier(in_features=x.shape[1], hidden=args.hidden, out_features=2, timesteps=args.timesteps).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in dl_train:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            accs = []
            for xb, yb in dl_test:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                accs.append(accuracy(logits, yb))

        print(f"epoch={epoch} test_acc={sum(accs)/len(accs):.4f}")


if __name__ == "__main__":
    main()
