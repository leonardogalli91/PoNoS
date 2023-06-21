import torch
from torch.utils.data import DataLoader


def get_metric_function(metric_name):
    if metric_name == "logistic_accuracy":
        return logistic_accuracy

    if metric_name == "softmax_accuracy":
        return softmax_accuracy

    elif metric_name == "softmax_loss":
        return softmax_loss

    elif metric_name == "softmax_loss_single":
        return softmax_loss_single

    elif metric_name == "logistic_loss":
        return logistic_loss

    elif metric_name == "logistic_loss_single":
        return logistic_loss_single

    elif metric_name == "squared_hinge_loss":
        return squared_hinge_loss

    elif metric_name == "mse":
        return mse_score

    elif metric_name == "squared_loss":
        return squared_loss

@torch.no_grad()
def compute_metric_on_dataset(model, dataset, metric_name, device):
    metric_function = get_metric_function(metric_name)
    
    model.eval()
    batch_size = 1024
    if device == "cuda":
        batch_size = 2048
    loader = DataLoader(dataset, drop_last=False, batch_size=batch_size, num_workers=6) #, pin_memory=True)
#    print("> Computing %s..." % (metric_name))

    score_sum = 0.
    #for images, labels in tqdm.tqdm(loader):
    for images, labels, indexes in loader:
        images, labels = images.to(device), labels.to(device)
        if model.parameters().__next__().dtype == torch.float16:
            images = images.half()
        elif model.parameters().__next__().dtype == torch.float64:
            images = images.double()

        score_sum += metric_function(model, images, labels).item() * images.shape[0] 
            
    score = float(score_sum / len(loader.dataset))

    return score

def softmax_loss(model, images, labels, backwards=False):
    logits = model(images)
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")
    loss = criterion(logits, labels.view(-1))

    if backwards and loss.requires_grad:
        loss.backward()

    return loss

def softmax_loss_single(model, images, labels, indexes, backwards=False):
    logits = model(images)
    crit = torch.nn.CrossEntropyLoss()

    loss = 0
    losses = []
    for log, lab in zip(logits, labels):
        single_loss = crit(torch.unsqueeze(log, 0), torch.unsqueeze(lab, 0))/images.shape[0]
        loss += single_loss
        losses.append(single_loss)
    if backwards and loss.requires_grad:
        loss.backward()

    return loss, losses, indexes.int()

def logistic_loss(model, images, labels, backwards=False):
    logits = model(images)
    criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")
    loss = criterion(logits.view(-1), labels.float().view(-1))

    if backwards and loss.requires_grad:
        loss.backward()

    return loss

#TODO: implement the other "single" losses (e.g., squared_loss_single)
def logistic_loss_single(model, images, labels, indexes, backwards=False):
    logits = model(images)
    criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")
    loss = 0
    losses = []
    for log, lab in zip(logits, labels):
        single_loss = criterion(log, torch.unsqueeze(lab, 0))/images.shape[0]
        loss += single_loss
        losses.append(single_loss)

    if backwards and loss.requires_grad:
        loss.backward()

    return loss, losses, indexes.int()

def squared_loss(model, images, labels, backwards=False):
    logits = model(images)
    criterion = torch.nn.MSELoss(reduction="mean")
    loss = criterion(logits.view(-1), labels.view(-1))

    if backwards and loss.requires_grad:
        loss.backward()

    return loss

def mse_score(model, images, labels):
    logits = model(images).view(-1)
    mse = ((logits - labels.view(-1))**2).float().mean()

    return mse

def squared_hinge_loss(model, images, labels, backwards=False):
    margin=1.
    logits = model(images).view(-1)

    y = 2*labels - 1

    loss = torch.mean((torch.max( torch.zeros_like(y) , 
                torch.ones_like(y) - torch.mul(y, logits)))**2 )

    if backwards and loss.requires_grad:
        loss.backward()

    return loss

def logistic_accuracy(model, images, labels):
    logits = torch.sigmoid(model(images)).view(-1)
    pred_labels = (logits > 0.5).float().view(-1)
    acc = (pred_labels == labels).float().mean()

    return acc

def softmax_accuracy(model, images, labels):
    logits = model(images)
    pred_labels = logits.argmax(dim=1)
    acc = (pred_labels == labels).float().mean()

    return acc
