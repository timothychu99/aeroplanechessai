"""Offline DQN trainer for Sorry! traces

This script loads `traces.jsonl` (one JSON record per line), converts each turn
into a (state, action, reward, next_state, done) transition, and trains a small
Q-network using fitted Q updates (offline DQN).

Usage:
    python3 ai_train.py --data traces.jsonl --epochs 50

If PyTorch is not installed the script will print instructions to install it.
"""

import json
import os
import argparse
from typing import List, Dict

MAX_POS = 58.0
NUM_PAWNS = 4
NUM_PLAYERS = 4 
ACTION_SIZE = 5  # pawn 1..4 => 0..3, pass/no-op => 4


def load_traces(path: str) -> List[Dict]:
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def state_vector_from_snapshot(snapshot: Dict, player_turn: int) -> List[float]:
    # snapshot keys: player_1_pawns, player_2_pawns, occupied_positions
    p1 = snapshot.get("player_1_pawns", [0]*NUM_PAWNS)
    p2 = snapshot.get("player_2_pawns", [0]*NUM_PAWNS)

    # normalize positions by MAX_POS
    vec = []
    # use current player as first block
    if player_turn == 0:
        vec.extend([pos / MAX_POS for pos in p1])
        vec.extend([pos / MAX_POS for pos in p2])
    else:
        vec.extend([pos / MAX_POS for pos in p2])
        vec.extend([pos / MAX_POS for pos in p1])

    # include player_turn scalar
    vec.append(float(player_turn))

    return vec


def build_transitions(records: List[Dict]):
    X = []
    A = []
    R = []
    Xp = []
    D = []

    for rec in records:
        s = rec.get("state_before")
        sa = rec.get("state_after")
        player_turn = rec.get("player_turn", 0)
        action = rec.get("action", -1)
        # map action: -1 -> 4 (pass), 1..4 -> 0..3
        if action == -1:
            a = 4
        else:
            a = int(action) - 1 if action > 0 else 4
            if a < 0 or a >= ACTION_SIZE:
                a = 4

        r = float(rec.get("reward", 0.0))

        x = state_vector_from_snapshot(s, player_turn)
        xp = state_vector_from_snapshot(sa, player_turn)

        # done if any pawn reached MAX_POS
        done = any([pos >= MAX_POS for pos in sa.get("player_1_pawns", [])]) or any([pos >= MAX_POS for pos in sa.get("player_2_pawns", [])])

        X.append(x)
        A.append(a)
        R.append(r)
        Xp.append(xp)
        D.append(1.0 if done else 0.0)

    return X, A, R, Xp, D


# Create the training script using PyTorch if available
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
except Exception:
    torch = None


if torch is None:
    def main():
        print("PyTorch is required to run this trainer.")
        print("Install with: python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")

else:
    class TransitionsDataset(Dataset):
        def __init__(self, X, A, R, Xp, D):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.A = torch.tensor(A, dtype=torch.int64)
            self.R = torch.tensor(R, dtype=torch.float32)
            self.Xp = torch.tensor(Xp, dtype=torch.float32)
            self.D = torch.tensor(D, dtype=torch.float32)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.A[idx], self.R[idx], self.Xp[idx], self.D[idx]


    class QNet(nn.Module):
        """ Multi-Layer Perceptron """
        def __init__(self, input_dim, action_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, action_dim)
            )

        def forward(self, x):
            return self.net(x)


    def train_dqn(X, A, R, Xp, D, device, epochs=20, batch_size=64, lr=1e-9, gamma=0.99, target_update=5, model_out="models/dqn.pth"):
        os.makedirs(os.path.dirname(model_out) or ".", exist_ok=True)

        dataset = TransitionsDataset(X, A, R, Xp, D)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        input_dim = len(X[0])
        qnet = QNet(input_dim, ACTION_SIZE).to(device)
        qtarget = QNet(input_dim, ACTION_SIZE).to(device)
        qtarget.load_state_dict(qnet.state_dict())

        opt = optim.Adam(qnet.parameters(), lr=lr)
        mse = nn.MSELoss()

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            for xb, ab, rb, xpb, db in loader:
                xb = xb.to(device)
                ab = ab.to(device)
                rb = rb.to(device)
                xpb = xpb.to(device)
                db = db.to(device)

                qvals = qnet(xb)
                qval = qvals.gather(1, ab.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    qnext = qtarget(xpb)
                    qnext_max, _ = qnext.max(dim=1)
                    target = rb + (1.0 - db) * gamma * qnext_max

                loss = mse(qval, target)
                opt.zero_grad()
                loss.backward()
                opt.step()
                epoch_loss += loss.item() * xb.size(0)

            epoch_loss /= len(dataset)
            if epoch % target_update == 0:
                qtarget.load_state_dict(qnet.state_dict())
            print(f"Epoch {epoch}/{epochs} loss={epoch_loss:.4f}")

        # save model
        torch.save({"model_state": qnet.state_dict(), "input_dim": input_dim}, model_out)
        print(f"Saved model to {model_out}")


    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument("--data", default="traces.jsonl")
        parser.add_argument("--epochs", type=int, default=20)
        parser.add_argument("--batch", type=int, default=64)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--gamma", type=float, default=0.99)
        parser.add_argument("--out", default="models/dqn.pth")
        args = parser.parse_args()

        print("Loading traces from", args.data)
        records = load_traces(args.data)
        if not records:
            print("No records found in", args.data)
            return

        X, A, R, Xp, D = build_transitions(records)

        if torch.cuda.is_available():
            # Set the device to the first GPU
            device = torch.device("cuda:0")
            print("CUDA is available! Using GPU.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(torch.version.cuda)
        print(device, "will be used for training")
        train_dqn(X, A, R, Xp, D, device, epochs=args.epochs, batch_size=args.batch, lr=args.lr, gamma=args.gamma, model_out=args.out)


if __name__ == "__main__":
    main()
