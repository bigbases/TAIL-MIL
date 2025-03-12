class AdditiveMIL(torch.nn.Module):
    def __init__(self, window, dim, sub_window):
        super(AdditiveMIL, self).__init__()
        
        seq_len = window // sub_window
        
        self.window = window
        self.dim = dim
        self.sub_window = sub_window
        self.seq_len = seq_len
        
        self.FetureExtracter1 = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv1d(
                    in_channels=1, out_channels=64, kernel_size=sub_window, stride=sub_window, padding=0
                ),
                torch.nn.LeakyReLU()
            )
            
            for i in range(dim)
        ])
        
        self.FetureExtracter2 = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv1d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(),
                torch.nn.MaxPool1d(3, stride=2),
                torch.nn.Flatten(),
                torch.nn.Linear(124, 64),
                torch.nn.LeakyReLU()
            )
            
            for i in range(dim)
        ])
        
        self.attention_V = torch.nn.Sequential(
            torch.nn.Linear(64, 128),
            torch.nn.Tanh()
        )

        self.attention_U = torch.nn.Sequential(
            torch.nn.Linear(64, 128),
            torch.nn.Sigmoid()
        )

        self.attention_weights = torch.nn.Linear(128, 1)
        
        self.lwv = torch.nn.Sequential(
            torch.nn.Linear(64, 32),
            torch.nn.LayerNorm(32),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(32, 2),
            torch.nn.LogSoftmax(dim=2)
        )
        
        self.weight_init(self)

    def weight_init(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight.data)

    def forward(self, x):
        embed = []
        wvs = []
        
        for i in range(self.dim):
            wv = []
            temp_x = x[:, i].reshape(x.shape[0], 1, -1)
            temp_x = self.FetureExtracter1[i](temp_x)
            temp_x = temp_x.transpose(1, 2)
            temp_x = temp_x.reshape(-1, 1, temp_x.shape[2])
            temp_x = self.FetureExtracter2[i](temp_x)
            temp_x = temp_x.reshape(x.shape[0], 1, self.seq_len, -1)
            embed.append(temp_x)
        
        embed = torch.cat(embed, 1)
        feature = embed.reshape(embed.shape[0], -1, 64)
        
        A_V = self.attention_V(feature)  # NxD
        A_U = self.attention_U(feature)  # NxD
        A = self.attention_weights(A_V * A_U).reshape(x.shape[0], -1, 1) # element wise multiplication # NxK
        A = torch.functional.F.softmax(A, dim=1)

        feature = feature *A
        wv = self.lwv(feature)
        output = torch.functional.F.log_softmax(torch.sum(wv, 1), dim=1)

        return (
            output, 
            wv.reshape(wv.shape[0], self.window, self.dim, 2).sum(2), 
            wv.reshape(wv.shape[0], self.window, self.dim, 2)
        )

class AttentionMIL(torch.nn.Module):
    def __init__(self, window, dim, sub_window):
        super(AttentionMIL, self).__init__()
        
        seq_len = window // sub_window
        
        self.window = window
        self.dim = dim
        self.sub_window = sub_window
        self.seq_len = seq_len
        
        self.FetureExtracter1 = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv1d(
                    in_channels=1, out_channels=64, kernel_size=sub_window, stride=sub_window, padding=0
                ),
                torch.nn.LeakyReLU()
            )
            
            for i in range(dim)
        ])
        
        self.FetureExtracter2 = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv1d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(),
                torch.nn.MaxPool1d(3, stride=2),
                torch.nn.Flatten(),
                torch.nn.Linear(124, 64),
                torch.nn.LeakyReLU()
            )
            
            for i in range(dim)
        ])
        
        self.attention_V = torch.nn.Sequential(
            torch.nn.Linear(64, 128),
            torch.nn.Tanh()
        )

        self.attention_U = torch.nn.Sequential(
            torch.nn.Linear(64, 128),
            torch.nn.Sigmoid()
        )

        self.attention_weights = torch.nn.Linear(128, 1)
        
        self.lwv = torch.nn.Sequential(
            torch.nn.Linear(64, 32),
            torch.nn.LayerNorm(32),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(32, 2),
            torch.nn.LogSoftmax(dim=1)
        )
        
        self.weight_init(self)

    def weight_init(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight.data)

    def forward(self, x):
        embed = []
        wvs = []
        
        for i in range(self.dim):
            wv = []
            temp_x = x[:, i].reshape(x.shape[0], 1, -1)
            temp_x = self.FetureExtracter1[i](temp_x)
            temp_x = temp_x.transpose(1, 2)
            temp_x = temp_x.reshape(-1, 1, temp_x.shape[2])
            temp_x = self.FetureExtracter2[i](temp_x)
            temp_x = temp_x.reshape(x.shape[0], 1, self.seq_len, -1)
            embed.append(temp_x)
        
        embed = torch.cat(embed, 1)
        feature = embed.reshape(embed.shape[0], -1, 64)
        
        A_V = self.attention_V(feature)  # NxD
        A_U = self.attention_U(feature)  # NxD
        A = self.attention_weights(A_V * A_U).reshape(x.shape[0], -1, 1) # element wise multiplication # NxK
        A = torch.functional.F.softmax(A, dim=1)

        feature = feature *A
        wv = self.lwv(feature.sum(1))
        
        outputs2 = (wv.unsqueeze(1).repeat(1, self.dim*self.window, 1)*A.reshape(A.shape[0], self.dim*self.window, 1))
        
        return wv, outputs2.reshape(x.shape[0], self.window, self.dim, 2).sum(2), outputs2.reshape(x.shape[0], self.window, self.dim, 2)
    
class MILNET(torch.nn.Module):
    def __init__(self, window, dim, sub_window):
        super(MILNET, self).__init__()
        
        seq_len = window // sub_window
        
        self.window = window
        self.dim = dim
        self.sub_window = sub_window
        self.seq_len = seq_len
        
        self.FetureExtracter1 = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv1d(
                    in_channels=1, out_channels=64, kernel_size=sub_window, stride=sub_window, padding=0
                ),
                torch.nn.LeakyReLU()
            )
            
            for i in range(dim)
        ])
        
        self.FetureExtracter2 = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv1d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(),
                torch.nn.MaxPool1d(3, stride=2),
                torch.nn.Flatten(),
                torch.nn.Linear(124, 64),
                torch.nn.LeakyReLU()
            )
            
            for i in range(dim)
        ])
        
        self.sent_gru = torch.nn.GRU(64, 32, bidirectional=True)
        self.dropout = torch.nn.Dropout(p=.2)
        
        self.attentionLayer1 = torch.nn.Linear(64, 32)
        self.attentionLayer2 = torch.nn.Linear(32, 1)
        
        self.lwv = torch.nn.Sequential(
            torch.nn.Linear(64, 32),
            torch.nn.LayerNorm(32),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(32, 2),
            torch.nn.LogSoftmax(dim=2)
        )
        
        self.weight_init(self)

    def weight_init(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight.data)

    def forward(self, x):
        embed = []
        wvs = []
        
        for i in range(self.dim):
            wv = []
            temp_x = x[:, i].reshape(x.shape[0], 1, -1)
            temp_x = self.FetureExtracter1[i](temp_x)
            temp_x = temp_x.transpose(1, 2)
            temp_x = temp_x.reshape(-1, 1, temp_x.shape[2])
            temp_x = self.FetureExtracter2[i](temp_x)
            temp_x = temp_x.reshape(x.shape[0], 1, self.seq_len, -1)
            embed.append(temp_x)
        
        embed = torch.cat(embed, 1)
        feature = embed.reshape(embed.shape[0], -1, 64)
        
        
        output_sent, state_sent = self.sent_gru(feature)

        sent_squish = self.attentionLayer1(output_sent)
        sent_squish = torch.sigmoid(sent_squish)
        sent_attn = self.attentionLayer2(sent_squish)
        sent_attn = torch.sigmoid(sent_attn)

        wv = self.lwv(feature)
        
        cont = sent_attn * wv
                
        output = torch.functional.F.log_softmax(torch.sum(cont, 1), dim=1)
        
        return output, cont.reshape(wv.shape[0], self.window, self.dim, 2).max(2).values, cont.reshape(wv.shape[0], self.window, self.dim, 2)
    
    
class miNet(torch.nn.Module):
    def __init__(self, window, dim, sub_window):
        super(miNet, self).__init__()
        
        seq_len = window // sub_window
        
        self.window = window
        self.dim = dim
        self.sub_window = sub_window
        self.seq_len = seq_len
        
        self.FetureExtracter1 = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv1d(
                    in_channels=1, out_channels=64, kernel_size=sub_window, stride=sub_window, padding=0
                ),
                torch.nn.LeakyReLU()
            )
            
            for i in range(dim)
        ])
        
        self.FetureExtracter2 = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv1d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(),
                torch.nn.MaxPool1d(3, stride=2),
                torch.nn.Flatten(),
                torch.nn.Linear(124, 64),
                torch.nn.LeakyReLU()
            )
            
            for i in range(dim)
        ])
                
        self.lwvs = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(64, 32),
                torch.nn.LayerNorm(32),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(32, 2),
                torch.nn.LogSoftmax(dim=1)
            )
            
            for i in range(dim)
        ])
        
        self.weight_init(self)

    def weight_init(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight.data)

    def forward(self, x):
        embed = []
        
        for i in range(self.dim):
            temp_x = x[:, i].reshape(x.shape[0], 1, -1)
            temp_x = self.FetureExtracter1[i](temp_x)
            temp_x = temp_x.transpose(1, 2)
            temp_x = temp_x.reshape(-1, 1, temp_x.shape[2])
            temp_x = self.FetureExtracter2[i](temp_x)
            temp_x = temp_x.reshape(x.shape[0], 1, self.seq_len, -1)
            embed.append(temp_x)
        
        embed = torch.cat(embed, 1)
        contributes1 = []
        output1 = torch.zeros((x.shape[0], self.seq_len, 2)).to(x.device)
        
        for i in range(self.seq_len):
            temp_x = embed[:, :, i]

            wv = []
            for j in range(self.dim):
                wv.append(self.lwvs[j](temp_x[:, j]).reshape(x.shape[0], 1, -1))
            wv = torch.cat(wv, 1)

            contributes1.append(wv.reshape(x.shape[0], 1, -1, 2))
                        
        contributes1 = torch.cat(contributes1, 1)
        
        output1 = contributes1.max(2).values
        output2 = contributes1.reshape(x.shape[0], -1, 2).max(1).values
        
        return output2, output1, contributes1
    
def positional_encoding(window_size, feature_dim):
    # Create the positional encodings matrix
    position = torch.arange(0, window_size, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, feature_dim, 2).float() * (-math.log(10000.0) / feature_dim))
    pe = torch.zeros(window_size, feature_dim)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)  # Add batch dimension
    return pe

class MILLET(torch.nn.Module):
    def __init__(self, window, dim, sub_window):
        super(MILLET, self).__init__()
        
        seq_len = window // sub_window
        
        self.window = window
        self.dim = dim
        self.sub_window = sub_window
        self.seq_len = seq_len
        
        self.FetureExtracter1 = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv1d(
                    in_channels=1, out_channels=64, kernel_size=sub_window, stride=sub_window, padding=0
                ),
                torch.nn.LeakyReLU()
            )
            
            for i in range(dim)
        ])
        
        self.FetureExtracter2 = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv1d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(),
                torch.nn.MaxPool1d(3, stride=2),
                torch.nn.Flatten(),
                torch.nn.Linear(124, 64),
                torch.nn.LeakyReLU()
            )
            
            for i in range(dim)
        ])
        
        self.attention_V = torch.nn.Sequential(
            torch.nn.Linear(64, 128),
            torch.nn.Tanh()
        )

        self.attention_U = torch.nn.Sequential(
            torch.nn.Linear(64, 128),
            torch.nn.Sigmoid()
        )

        self.attention_weights = torch.nn.Linear(128, 1)
        
        self.lwv = torch.nn.Sequential(
            torch.nn.Linear(64, 32),
            torch.nn.LayerNorm(32),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(32, 2),
            torch.nn.LogSoftmax(dim=2)
        )
        
        self.weight_init(self)

    def weight_init(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight.data)
            
    def forward(self, x):
        embed = []
        wvs = []
        
        for i in range(self.dim):
            wv = []
            temp_x = x[:, i].reshape(x.shape[0], 1, -1)
            temp_x = self.FetureExtracter1[i](temp_x)
            temp_x = temp_x.transpose(1, 2)
            temp_x = temp_x.reshape(-1, 1, temp_x.shape[2])
            temp_x = self.FetureExtracter2[i](temp_x)
            temp_x = temp_x.reshape(x.shape[0], 1, self.seq_len, -1)
            embed.append(temp_x)
        
        feature = torch.cat(embed, 1)
        pe = positional_encoding(self.window, 64).to(x.device) 
        feature = feature + pe.reshape(1, pe.shape[0], pe.shape[1], pe.shape[2])
        feature = feature.reshape(feature.shape[0], -1, 64)
        
        A_V = self.attention_V(feature)  # NxD
        A_U = self.attention_U(feature)  # NxD
        A = self.attention_weights(A_V * A_U).reshape(x.shape[0], -1, 1) # element wise multiplication # NxK
        A = torch.functional.F.softmax(A, dim=1)

        wv = self.lwv(feature)
        
        cont = torch.mul(wv, A)
                
        output = torch.functional.F.log_softmax(torch.sum(cont, 1), dim=1)
        
        return output, cont.reshape(wv.shape[0], 12, 5, 2).max(2).values, cont.reshape(wv.shape[0], 12, 5, 2)