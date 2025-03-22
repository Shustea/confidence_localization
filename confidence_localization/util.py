import torch
import matplotlib.pyplot as plt
from numpy import unique

def save_sample_as_image(tensor: torch.Tensor, label: torch.Tensor, filename: str, path='/workspaces/confidence_localization/samples/'):
    # Ensure tensor is on CPU and detach if it's a computation graph tensor
    if tensor.is_cuda:
        tensor = tensor.cpu()
    tensor = tensor.detach()

    plt.figure()
    plt.imshow(tensor.numpy().T, origin='lower')
    plt.axis("off")
    plt.colorbar()

    plt.title(f'GT is at : {str(unique(label[~label.isnan()].cpu()))} [radians]', fontsize=14, fontweight="bold")
    plt.xlabel('Time Frame')
    plt.ylabel('Frequency Bins')

    # Save the image
    plt.savefig(path + filename, bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close()

def save_doas(tensor: torch.Tensor, label: torch.Tensor, filename: str, path='/workspaces/confidence_localization/samples/'):
    # Ensure tensor is on CPU and detach if it's a computation graph tensor
    if tensor.is_cuda:
        tensor = tensor.cpu()
    tensor = tensor.detach()

    plt.figure()
    plt.hist(tensor.numpy(), bins=20)

    plt.xlabel('DOA result [radians]')
    plt.ylabel('Count #')

    plt.title(f'distribiution for case where GT is at : {str(unique(label[~label.isnan()].cpu()))} [radians]', fontsize=14, fontweight="bold")

    # Save the image
    plt.savefig(path + filename, bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close()