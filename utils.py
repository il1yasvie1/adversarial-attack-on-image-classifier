import torch
import numpy as np
import matplotlib.pyplot as plt
from cifar10_resnet20 import classes
from tqdm import tqdm
import time
import pandas as pd


def img2np(img):
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std  = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
    img = img * std + mean
    img = torch.clip(img, 0, 1)
    return np.transpose(img.numpy(), (1, 2, 0))


def plot_examples(dataset):
	classes = dataset.classes
	num_classes = len(classes)
	examples_dict = {}
	for idx in range(len(dataset)):
		img, label = dataset[idx]
		label = int(label)
		if label not in examples_dict:
			img_np = img2np(img)
			examples_dict[label] = img_np
		if len(examples_dict) == num_classes:
			break
	fig, axes = plt.subplots(1, num_classes, figsize=(8 * num_classes, 8))
	for idx, (label, img) in enumerate(examples_dict.items()):
		axes[idx].imshow(img)
		axes[idx].set_title(classes[label])
		axes[idx].axis('off')
	plt.show()


def calc_prob(model, x0, y):
  logits = model(x0.unsqueeze(0))
  p = torch.softmax(logits, dim=1)[0][y].item()
  return p


def show_adversarial_example(model, x0, y, d):
    x_adv = x0 + d
    with torch.no_grad():
        logits = model(x_adv.unsqueeze(0))
        probs = torch.softmax(logits, dim=1)
        yh = torch.argmax(probs, dim=1).item()
        p_adv = probs[0, yh].item()

    p_orig = calc_prob(model, x0, y)

    fig, axes = plt.subplots(1, 3, figsize=(10, 5))

    axes[0].imshow(img2np(x0))
    axes[0].set_title(f'Original: {classes[y]}\nconf. = {p_orig:.3f}')
    axes[1].imshow(img2np(x_adv))
    axes[1].set_title(f'Adversarial: {classes[yh]}\nconf. = {p_adv:.3f}')
    pert_vis = torch.clip(d * 100 + 0.5, 0, 1)
    axes[2].imshow(np.transpose(pert_vis.numpy(), (1, 2, 0)))
    axes[2].set_title('Perturbation ($\\times 100$)')

    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def accuracy(model, testloader):
    num_classes = len(testloader.dataset.classes)
    correct_per_class = torch.zeros(num_classes)
    total_per_class = torch.zeros(num_classes)
    model.eval()
    with torch.no_grad():
        for images, labels in testloader:
            outputs = model(images)
            predicted = outputs.argmax(1)
            total_per_class += torch.bincount(labels, minlength=num_classes)
            mask = (predicted == labels)
            correct_per_class += torch.bincount(labels[mask], minlength=num_classes)

    accs = 100 * correct_per_class / total_per_class.clamp(min=1)

    for name, acc in zip(testloader.dataset.classes, accs):
        print(f"Accuracy for class '{name}': {acc:.2f}%")

    print(f"\nOverall accuracy: {100 * correct_per_class.sum() / total_per_class.sum():.2f}%")


def get_correct_indices(model, loader, num_per_class=64):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            all_preds.append(outputs.argmax(1).cpu())
            all_labels.append(labels.cpu())
    
    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    
    correct_mask = (preds == labels)
    all_indices = torch.arange(len(labels)) 
    
    final_indices = []
    num_classes = labels.max().item() + 1
    
    for i in range(num_classes):
        class_mask = (labels == i) & correct_mask
        class_indices = all_indices[class_mask]

        final_indices.extend(class_indices[:num_per_class].tolist())
        
    return final_indices


def attack_success_rate(dataloader, model, attack, e, p, N):
    all_success = []
    file_name = f'{attack.__name__}-e{e}-p{p}-N{N}.csv'
    
    results = {
         'l1': [], 'l2': [], 'linf': [],
         'times': [],
         'prob_org': [], 'prob_adv': []
    }

    for x0, y in tqdm(dataloader, desc="Attacking"):
        batch_size = x0.size(0)

        with torch.no_grad():
            orig_logits = model(x0)
            orig_probs = torch.nn.Softmax(dim=1)(orig_logits)
   
            p_orig = orig_probs.gather(1, y.view(-1, 1)).squeeze()


        t0 = time.time()
        d = attack(model, x0, y, e, p, N)

        t = (time.time() - t0) / batch_size 

  
        with torch.no_grad():
            adv_logits = model(x0 + d)
            yh = torch.nn.Softmax(dim=1)(adv_logits) 
            
            preds = yh.argmax(dim=1)
            success = (preds != y)
            all_success.append(success.cpu())
            
            p_adv = yh.gather(1, y.view(-1, 1)).squeeze()
            
            d_flat = d.flatten(1)
            l1 = d_flat.norm(1, dim=1)
            l2 = d_flat.norm(2, dim=1)
            linf = d_flat.norm(float('inf'), dim=1)

        results['prob_org'].extend(p_orig.view(-1).cpu().tolist())
        results['prob_adv'].extend(p_adv.view(-1).cpu().tolist())
        results['l1'].extend(l1.cpu().tolist())
        results['l2'].extend(l2.cpu().tolist())
        results['linf'].extend(linf.cpu().tolist())
        results['times'].extend([t] * batch_size) 

    all_success = torch.cat(all_success)
    successful_indices = all_success.nonzero(as_tuple=True)[0].tolist()
    success_rate = all_success.float().mean().item()

    print(f"\nFinal Rate: {success_rate:.2%}")
    
    df = pd.DataFrame(results)
    df['success'] = all_success.numpy()
    df['global_index'] = df.index
    df.to_csv(file_name, index=False)
    print(f"Results saved to {file_name}")

    return success_rate, successful_indices
