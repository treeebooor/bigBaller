import torch
import torch.nn as nn
import torch.nn.functional as F
import math


#encoding class
#encode players coords and ball
class PlayersBallEncoding(nn.Module):
    '''
    Player and Ball encoding Network
    '''
    def __init__(self,input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        out = F.relu(out)
        out = self.fc4(out)
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    
class BallerModel(nn.Module):
    def __init__(self, 
                 player_ball_encoding_size,
                 dropout=0.2):
        super().__init__()
        
        # 1*(xyz)
        self.ball_encoding = PlayersBallEncoding(
            3,
            player_ball_encoding_size
        )
        
        # 10*(xy)
        self.players_encoding = PlayersBallEncoding(
            20,
            player_ball_encoding_size
        )
        
        # 1*(xyz)
        self.tgt_encoding = PlayersBallEncoding(
            3,
            player_ball_encoding_size*2
        )
        
        self.pos_encoder = PositionalEncoding(
            d_model=player_ball_encoding_size*2,
            dropout=dropout
        )
        
        #d_model is player_ball_encoding_size*2 because we concatenate ball and player encodings
        encoderLayer = nn.TransformerEncoderLayer(
            d_model=player_ball_encoding_size*2,
            nhead=8,
            dropout=dropout
        )   #input and output is (seq,batch, feature)
        
        self.transformerEncoder = nn.TransformerEncoder(
            encoder_layer=encoderLayer,
            num_layers=6
        )
        
        decoderLayer = nn.TransformerDecoderLayer(
            d_model=player_ball_encoding_size*2,
            nhead=8,
            dropout=dropout
        )

        self.transformerDecoder = nn.TransformerDecoder(
            decoder_layer=decoderLayer,
            num_layers=6
        )
        
        self.linearMapping = nn.Linear(
            in_features=player_ball_encoding_size*2,
            out_features=3
        ) # (xyz) of ball
        
    def forward(self, x, tgt, src_mask=None, tgt_mask=None):
        ball_pos = x[:,:,0,:] #shape (batch, seq_len, 3)
        ball_pos_encoded = self.ball_encoding(ball_pos) #shape (batch, seq_len, encoding_dim)
        
        player_pos = x[:,:,1:,:-1] #shape (batch, seq_len, 10, 2)
        b,s,_,_ = player_pos.shape
        # stack all player positions
        player_pos = torch.reshape(player_pos, (b,s,-1))#shape (batch, seq_len, 20)

        player_pos_encoded = self.players_encoding(player_pos) #shape (batch, seq_len, encoding dim)
        
        combined = torch.cat((ball_pos_encoded, player_pos_encoded), dim=-1)  #should be shape (batch,seq,d_model)
        combined = torch.permute(combined, (1,0,2)) # should be shape (seq, batch,d_model)
        
        combined = self.pos_encoder(combined)
        
        src = self.transformerEncoder(combined) #shape (seq, batch,d_model)
        
        decoder_output = self.tgt_encoding(tgt) #shape (batch, out_seq_len, 2*encoding_dim)
        decoder_output = torch.permute(decoder_output, (1,0,2)) # should be shape (seq, batch,d_model)
        
        decoder_output = self.transformerDecoder(
            tgt=decoder_output,
            memory=src,
            tgt_mask=tgt_mask,
            memory_mask=src_mask
        )
        
        prediction = self.linearMapping(decoder_output).permute((1,0,2))
        
        return prediction
        
    def generate_square_subsequent_mask(dim1, dim2):
        return torch.triu(torch.full((dim1, dim2), float('-inf')), diagonal=1).double()