class attention(nn.Module)
	def __init__(self, 
				dim,
				num_head,
				dropout):
		super(attention,self).__init__()
		self.dim = dim
		self.num_head = num_head
        self.head_dim = dim // num_head
		self.dropout = dropout
		self.q = nn.Linear(dim, dim, bias = True) #dim x dim
		self.k = nn.Linear(dim, dim, bias = True) #dim x dim
		self.v = nn.Linear(dim, dim, bias = True) #dim x dim
		self.proj = nn.Linear(dim, dim)
  
	def forward(x, mask):
        batch_size, seq_len, dim = x.size()
        Q = self.q(x) # batch x seq_len x dim
        K = self.k(x) # batch x seq_len x dim
        V = self.v(x) # batch x seq_len x dim
        Q = Q.view(batch_size, self,num_head, seq_len, self.head_dim)
        K = K.view(batch_size, self,num_head, seq_len, self.head_dim)
        V = V.view(batch_size, self,num_head, seq_len, self.head_dim)
        
        attn = torch.matmul(Q, K.transpose(-2, -1))/ torch.sqrt(torch.tensor(self.head_dim))# batch x num_head x seq_len x seq_len
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn_weight = F.softmax(attn, dim = -1)
        attn_weight = self.dropout(attn_weight)
        
        
            
        #get the attention value
        attn_value = torch.matmul(attn, V).view(batch_size, seq_len, dim)
        
        #to the FFN
        output = self.proj(attn_value)
        return output, attn_weight
        