import torch
import torch.nn as nn
import torchvision.models as models
import torch.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN,self).__init__()
        self.embed_size=embed_size
        self.vocab_size=vocab_size
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.Linear = nn.Linear(hidden_size, vocab_size)
        self.sm = nn.Softmax()
    
    def forward(self, features, captions):
        # Get caption embedding
        embeds=self.word_embedding(captions[:,:-1])
        #Concatenate the feature embedding with caption embedding
        concat_embeds = torch.cat((features.view(-1, 1,self.embed_size), embeds),1)
        #Pass the embeds to the rnn
        output, _ = self.lstm(concat_embeds)
        #Pass the output to Linear layer to get the output as the vocab size
        output=self.Linear(output)
#         output=output.argmax(2)
#         output = output[:,1:,:]
#         output = self.sm(output)
        return output

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        sentense = []
        output = inputs
        h=torch.zeros((1,1,512)).to(device)
        c=h
        for i in range(max_len):
            output, states = self.lstm(output,states)
            output = self.Linear(output)
            word_id = output.argmax(dim=2)
            sentense.append(word_id.item())
            output = self.word_embedding(word_id).view(1,1,-1)
            
        return sentense            
            