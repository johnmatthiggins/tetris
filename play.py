#!/usr/bin/env python3
import sys
import torch
import numpy as np

from gameboy import GBGym
from tetris import TetrisNN

# if GPU is to be used
device = "cpu"

if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print(
            "MPS not available because the current PyTorch install was not "
            "built with MPS enabled."
        )
    else:
        print(
            "MPS not available because the current MacOS version is not 12.3+ "
            "and/or you do not have an MPS-enabled device on this machine."
        )

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"

torch.device(device)


def main():
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    else:
        model_name = "model.pt"

    print("loading %s" %(model_name))
    gb = GBGym(device=device, speed=1)
    model = TetrisNN(len(gb.action_space)).to(device)

    # load weights...
    model.load_state_dict(torch.load(model_name))
    model.eval()

    state, info = gb.reset()
    state = state.unsqueeze(0)
    terminated = False

    while not terminated:
        next_action = model(state).max(1)[1]
        state, reward, terminated, truncated = gb.step(next_action.item())
        state = state.unsqueeze(0)

    gb.close()


if __name__ == "__main__":
    main()
