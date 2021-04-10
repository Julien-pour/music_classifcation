from tqdm.notebook import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torchaudio import transforms
from torchvision import transforms as T

import numpy as np
#dataset


from torch.utils.data import Dataset, DataLoader

# creation of torch dataset


#type_set= train or test 
#audio_ids
label_dic = {'Electronic': 0,  'Experimental':1, 'Folk':2,'Hip-Hop':3,
 'Instrumental':4,
 'International':5,
 'Pop':6,
 'Rock':7}


class Audio_classification(Dataset):
    def __init__(self, type_set, audio_idx, class_ids, label_dic, audio_resize, load_n_sec_audio=15, transform = None):#("train",train,data,)
            self.type_set=type_set
            self.audio_ids = audio_idx
            self.class_ids = class_ids
            self.label_dic = label_dic
            self.time_audio_resize=audio_resize
            self.transform = transform
            self.load_n_sec_audio=load_n_sec_audio
                
    def __len__(self):
          return len(self.audio_ids)
    
    def __getitem__(self, index):
            #on prend 3s
            load_first_n_sec_audio=self.load_n_sec_audio
            sample_rate_default=22050
            #time_audio_resize #= 3#sec
            
            #audio_path is like 000002.wav but self.audio_ids[index] has not all the 0 include so ...
            idx=self.audio_ids[index]
            count=len(str(idx))
            #print("The number of digits in the number are:",count)
            if count!=6:
                idx="0"*(6-count)+str(idx)
            audio_path=str(idx)

            filename = "data_proces/fma_small/"+ audio_path +".wav"
            label = self.label_dic[self.class_ids.loc[self.audio_ids[index]]]
            
            #load audio mp3
            
            waveform, sample_rate=torchaudio.load(filename, normalize = True)
            #waveform, sample_rate=sf.read(filename,sr=None)
                
            len_audio_resize = int(sample_rate_default * self.time_audio_resize)
            indice_max=len(waveform[0])-len_audio_resize-1
            indice_initial=np.random.randint(max(indice_max,1))
            #make time_audio_resize second long audio 
            if self.type_set=="train":
                waveform = waveform[:,indice_initial:indice_initial+len_audio_resize]
            else:
                waveform = waveform[:,0:len_audio_resize]
                
            # if weveform is to short zero padding at the end 
            if waveform.size()!=(1,len_audio_resize):
                waveform=torch.cat((waveform,torch.zeros((1,len_audio_resize-waveform.size(1)))),1)

            if self.transform:
                waveform = self.transform(waveform)
            return waveform,torch.tensor(label, dtype = torch.float)

bs=16

#init param
# init mel, mfcc, spectrogramme for model input size = 384 or 224
def init_param(model, train_index, test_index, data, bs, label_dic,device=device):
    if model.model.image_size==(384,384):
        print("384")
        #audio_resize=8.9
        trainset  = Audio_classification("train", train_index, data, label_dic, 8.9)
        trainloader  = DataLoader(trainset , batch_size=bs,
                                shuffle=True, num_workers=0)

        testset = Audio_classification("test", test_index, data, label_dic, 8.9)
        testloader   = DataLoader(testset , batch_size=bs,
                            shuffle=False, num_workers=0)

        sample_rate=22050
        n_fft = 2048
        win_length = None
        hop_length = 512
        n_mels = 512
        n_mfcc = 384
        log_mels=True
        mfcc_transform = transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,log_mels=log_mels, melkwargs={'n_fft': n_fft, 'n_mels': n_mels, 'hop_length': hop_length})


        n_fft = 2048
        win_length = None
        hop_length = 512
        n_mels = 384#128
        sample_rate=22050
        mel_spectrogram = transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm='slaney',
            onesided=True,
            n_mels=n_mels,
        )


        n_fft = 766
        win_length = None
        hop_length = 512

        # define transformation
        spectrogram = transforms.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            center=True,
            pad_mode="reflect",
            power=2.0,
            normalized=True #a test
        )

        normalize_spec_db = T.Normalize(mean=[-15.0580, -63.8859, -43.7740],
                            std=[19.4417, 48.3987, 21.0964])


    if model.model.image_size==(224,224):
        print("224")
        

        audio_resize_2=5.2
        trainset  = Audio_classification("train",train_index,data,label_dic,audio_resize_2)
        trainloader  = DataLoader(trainset , batch_size=bs,
                                shuffle=True, num_workers=0)

        testset = Audio_classification("test",test_index,data,label_dic,audio_resize_2)
        testloader   = DataLoader(testset , batch_size=bs,
                                shuffle=False, num_workers=0)
        sample_rate=22050
        n_fft = 2048
        win_length = None
        hop_length = 512
        n_mels = 256
        n_mfcc = 224
        log_mels=True
        mfcc_transform = transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,log_mels=log_mels, melkwargs={'n_fft': n_fft, 'n_mels': n_mels, 'hop_length': hop_length})


        n_fft = 2048
        win_length = None
        hop_length = 512
        n_mels = 224#128
        sample_rate=22050
        mel_spectrogram = transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm='slaney',
            onesided=True,
            n_mels=n_mels,
        )


        n_fft = 446
        win_length = None
        hop_length = 512

        # define transformation
        spectrogram = transforms.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            center=True,
            pad_mode="reflect",
            power=2.0,
        )
        normalize_spec_db = T.Normalize(mean=[-13.8637, -53.9947, -21.2606],
                            std=[19.2690, 49.1503, 22.0783])
    
    all_spectro=[mel_spectrogram.to(device),mfcc_transform.to(device),spectrogram.to(device)]

    return normalize_spec_db, all_spectro, trainloader, testloader


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

# plot spectrogram


def plot_spectrogram(spec, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
  fig, axs = plt.subplots(1, 1)
  axs.set_title(title or 'Spectrogram (db)')
  axs.set_ylabel(ylabel)
  axs.set_xlabel('frame')
  im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)#librosa.power_to_db(spec)
  if xmax:
    axs.set_xlim((0, xmax))
  fig.colorbar(im, ax=axs)
  plt.show(block=False)