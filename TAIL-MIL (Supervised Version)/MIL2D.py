import torch

class TAILMIL(torch.nn.Module):
    def __init__(self, window, dim, sub_window):
        super(TAILMIL, self).__init__()
        
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
        
        self.BeforeInstanceEncoder = torch.nn.Sequential(
            torch.nn.Linear(64, 64),
            torch.nn.LayerNorm(64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.LeakyReLU()
        )
        
        self.lwq = torch.nn.Sequential(
            torch.nn.Linear(64, 1),
            torch.nn.Sigmoid()
        )
        
        self.lwk = torch.nn.Sequential(
            torch.nn.Linear(64, dim),
            torch.nn.Sigmoid()
        )
        
        self.lwvs = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(64, 32),
                torch.nn.LayerNorm(32),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(32, 1),
                torch.nn.ReLU()
            )
            
            for i in range(dim)
        ])
        
        self.lwq2 = torch.nn.Sequential(
            torch.nn.Linear(64*dim, 1),
            torch.nn.Sigmoid()
        )
        
        self.lwk2 = torch.nn.Sequential(
            torch.nn.Linear(64*dim, seq_len),
            torch.nn.Sigmoid()
        )
        
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
        output1 = torch.zeros((x.shape[0], self.seq_len, 1)).to(x.device)
        
        for i in range(self.seq_len):
            if i == 0:
                acmlF = self.BeforeInstanceEncoder(embed[:, :, i])
                
            else:
                acmlF = acmlF + self.BeforeInstanceEncoder(embed[:, :, i])
            
            temp_x = embed[:, :, i]
            wq = self.lwq(temp_x)
            wk = self.lwk(acmlF)
            wv = []
            for j in range(self.dim):
                wv.append(self.lwvs[j](temp_x[:, j]).reshape(wq.shape[0], 1, -1))
            wv = torch.cat(wv, 1)
                
            att1 = torch.functional.F.softmax(
                torch.matmul(
                    wq.transpose(1, 2), 
                    (wk.transpose(1, 2)) / torch.sqrt(torch.tensor(self.dim))
                ), 2
            ).reshape(x.shape[0], -1, 1)
            
            cont = torch.mul(wv, att1)
            contributes1.append(cont.reshape(x.shape[0], 1, -1, 1))

            output1[:, i] = torch.sum(cont, 1)
            
        contributes1 = torch.cat(contributes1, 1)
        embed = embed.transpose(1, 2)
        wq_a = self.lwq2(embed.reshape(x.shape[0], self.seq_len, -1))
        wk_a = self.lwk2(embed.reshape(x.shape[0], self.seq_len, -1))    
        att2 = torch.functional.F.softmax(torch.matmul(wq_a.transpose(1, 2), wk_a.transpose(1, 2)) / torch.sqrt(torch.tensor(self.window)), 2)
        contributes2 = torch.mul(output1, att2.reshape(-1, self.seq_len, 1))

        output2 = torch.sum(contributes2, 1)
        
        return output2, contributes2, contributes1
    

class NestedMIL(torch.nn.Module):
    def __init__(self, window, dim, sub_window):
        super(NestedMIL, self).__init__()
        
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
        
        self.attention_V2 = torch.nn.Sequential(
            torch.nn.Linear(64, 128),
            torch.nn.Tanh()
        )

        self.attention_U2 = torch.nn.Sequential(
            torch.nn.Linear(64, 128),
            torch.nn.Sigmoid()
        )

        self.attention_weights2 = torch.nn.Linear(128, 1)
        
        self.lwv = torch.nn.Sequential(
            torch.nn.Linear(64, 32),
            torch.nn.LayerNorm(32),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(32, 1),
            torch.nn.ReLU()
        )
        
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
        
        A1 = []
        embedes = []
        for i in range(self.seq_len):
            temp_x = embed[:, :, i]
            
            A_V = self.attention_V(temp_x)  # NxD
            A_U = self.attention_U(temp_x)  # NxD
            A = self.attention_weights(A_V * A_U).reshape(x.shape[0], -1, 1) # element wise multiplication # NxK
            A = torch.functional.F.softmax(A, dim=1)
        
            embedes.append((temp_x * A).reshape(x.shape[0], 1, self.dim, 64))

            A1.append(A.reshape(x.shape[0], 1, -1))
        
        A1 = torch.concat(A1, 1)
        embedes = torch.concat(embedes, 1)

#         return embedes
        features = embedes.mean(2)
        
        A_V = self.attention_V2(features)  # NxD
        A_U = self.attention_U2(features)  # NxD
        A = self.attention_weights2(A_V * A_U).reshape(x.shape[0], -1, 1) # element wise multiplication # NxK
        A = torch.functional.F.softmax(A, dim=1)
        
        features = features * A
        
        features = features.mean(1)
        
        output1 = self.lwv(features)
        output2 = (output1.unsqueeze(1).repeat(1, self.window, 1)*A.reshape(A.shape[0], self.window, 1))
        output3 = (output2.unsqueeze(2).repeat(1, 1, self.dim, 1)*A1.reshape(A1.shape[0], self.window, self.dim, 1))
        return output1, output2, output3