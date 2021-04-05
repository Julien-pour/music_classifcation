from tqdm.notebook import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio

#train test  function

#https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html
use_amp = False #mixed precision float 16
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

def train_transf(model, device, trainloader, criterion, optimizer, epoch,scheduler,all_spectro,transform_train_db, log_interval,grad_clip=None,Use_waveform=False):
    model.train()
    correct = 0
    train_loss = 0
    use_waveform=Use_waveform
    for batch_idx, (data, target) in enumerate(tqdm(trainloader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        with torch.no_grad():
            spec=torch.tensor([]).to(device)
            if use_waveform:#use waveform only
                #all_spectro=all_spectro[1:]
#                 wave=data[:,:,:transformer.image_size[0]**2]
#                 data=wave.view(wave.size(0),1,transformer.image_size[0],transformer.image_size[0]).repeat(1,3,1,1)
                im=data
                size_in=im.size(2)
                zero_pad=torch.zeros(im.size(0),1,model.image_size[0]**2*3-size_in).to(device)
                concat=torch.cat((zero_pad,im),2).view(im.size(0),3,model.image_size[0],model.image_size[0])
                data=concat
            else:
                for spectro in all_spectro :
                    spec1=spectro(data)
                    spec=torch.cat((spec,spec1),1)
                spec=spec.to(device)
                spec=transform_train_db(spec)
                data=spec.to(device)
                #data=torch.cat((spec,data[:,:transformer.image_size[0]**2]),1)
        with torch.cuda.amp.autocast(enabled=use_amp): # full mixed precision enabled=use_amp   
            output = model(data).squeeze(0)
            loss = criterion(output, target.long())
        scaler.scale(loss).backward()
        if grad_clip: 
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()
        
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
#         loss.backward()
#         optimizer.step()
        train_loss += loss.item()
#     if batch_idx % log_interval == 0:
    print('Train Epoch: {}  \tLoss: {:.6f}   Accuracy: {}/{} ({:.0f}%'.format(
        epoch, loss.item(), correct, len(trainloader.dataset), 100. * correct / len(trainloader.dataset)))
    return train_loss/len(trainloader.dataset),100. * correct / len(trainloader.dataset)


def test_transf(model, device, testloader,criterion,all_spectro,transform_test_db,Use_waveform=False):
    use_waveform=Use_waveform
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            if use_waveform:#use waveform only
                #all_spectro=all_spectro[1:]
#                 wave=data[:,:,:transformer.image_size[0]**2]
#                 data=wave.view(wave.size(0),1,transformer.image_size[0],transformer.image_size[0]).repeat(1,3,1,1)
                im=data
                size_in=im.size(2)
                zero_pad=torch.zeros(im.size(0),1,model.image_size[0]**2*3-size_in).to(device)
                concat=torch.cat((zero_pad,im),2).view(im.size(0),3,model.image_size[0],model.image_size[0])
                data=concat
            else:
                spec=torch.tensor([]).to(device)
                for spectro in all_spectro :
                    spec1=spectro(data)
                    spec=torch.cat((spec,spec1),1)
                data=spec.to(device)
                data=transform_test_db(data)
            output = model(data.to(device)).squeeze(0)
            test_loss += criterion(output, target.long()).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(testloader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(testloader.dataset),
        100. * correct / len(testloader.dataset)))
    return test_loss,100. * correct / len(testloader.dataset)