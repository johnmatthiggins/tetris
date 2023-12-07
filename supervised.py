import torch

from tetris import TetrisNN

def main():
    model = TetrisNN(len(gb.action_space)).to(device)

    # load weights...
    model.load_state_dict(torch.load("model.pt"))
    model.eval()

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)

if __name__ == '__main__':
    main()
