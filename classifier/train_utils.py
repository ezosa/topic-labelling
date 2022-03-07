
import torch


def save_checkpoint(save_path, model, optimizer, valid_loss):
    if save_path == None:
        return

    state_dict = {'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'valid_loss': valid_loss}

    torch.save(state_dict, save_path)
    print('Model saved to ==>', save_path)


def load_checkpoint(load_path, model, optimizer, device):
    if load_path == None:
        return

    state_dict = torch.load(load_path, map_location=device)
    print('Model loaded from <== ', load_path)

    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])

    return state_dict['valid_loss']
