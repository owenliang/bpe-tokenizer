from collections import OrderedDict
import pickle
from tqdm import tqdm

# Byte-Pair Encoding tokenization
class BPETokenizer:
    def __init__(self):
        self.b2i=OrderedDict() # bytes to ids
        self.i2b=OrderedDict() # ids to bytes (b2i的反向映射)
        self.next_id=0
    
    # 相邻token统计
    def _pair_stats(self,tokens,stats):
        for i in range(len(tokens)-1):
            new_token=tokens[i]+tokens[i+1]
            if new_token not in stats:
                stats[new_token]=0
            stats[new_token]+=1
            
    # 合并相邻token
    def _merge_pair(self,tokens,new_token):
        merged_tokens=[]
        
        i=0
        while i<len(tokens):
            if i+1<len(tokens) and tokens[i]+tokens[i+1]==new_token:
                merged_tokens.append(tokens[i]+tokens[i+1])
                i+=2
            else:
                merged_tokens.append(tokens[i])
                i+=1
        return merged_tokens
    
    def train(self,text_list,vocab_size):
        # 单字节是最基础的token，初始化词表
        for i in range(256):
            self.b2i[bytes([i])]=i
        self.next_id=256
        
        # 语料转byte
        tokens_list=[]
        for text in text_list:
            tokens=[bytes([b]) for b in text.encode('utf-8')]
            tokens_list.append(tokens)
        
        # 进度条
        progress=tqdm(total=vocab_size-256)
        
        while True:
            # 词表足够大了，退出训练
            if self.next_id>=vocab_size:
                break
            
            # 统计相邻token频率
            stats={}
            for tokens in tokens_list:
                self._pair_stats(tokens,stats)

            # 没有更多相邻token, 无法生成更多token，退出训练
            if not stats:   
                break 
            
            # 合并最高频的相邻token,作为新的token加入词表
            new_token=max(stats,key=stats.get)

            new_tokens_list=[]
            for tokens in tokens_list:
                new_tokens_list.append(self._merge_pair(tokens,new_token))
            tokens_list=new_tokens_list

            # new token加入词表
            self.b2i[new_token]=self.next_id
            self.next_id+=1
            
            # 刷新进度条
            progress.update(1)
        
        self.i2b={v:k for k,v in self.b2i.items()}

    # 词表大小
    def vocab_size(self):
        return self.next_id
    
    # 词表
    def vocab(self):
        return self.b2i
    
    def encode(self,text):
        tokens=[bytes([b]) for b in text.encode('utf-8')]
        while True:
            # 统计相邻token频率
            stats={}
            self._pair_stats(tokens,stats)
            
            # 选择id最为的pair合并（也就是优先合并短的）
            new_token=None
            for merge_token in stats:
                if merge_token in self.b2i and (new_token is None or self.b2i[merge_token]<self.b2i[new_token]):
                    new_token=merge_token
            
            # 没有可以合并的pair，退出
            if new_token is None:
                break

            # 合并pair
            tokens=self._merge_pair(tokens,new_token)
        return [self.b2i[tok] for tok in tokens],tokens
    
    def decode(self,tokens):
        return b''.join([self.i2b[id] for id in tokens]).decode('utf-8',errors='replace')
    
    def save(self,file):
        with open(file,'wb') as fp:
            fp.write(pickle.dumps(self.b2i))
    
    def load(self,file):
        with open(file,'rb') as fp:
            self.b2i=pickle.loads(fp.read())
        self.next_id=len(self.b2i)
        self.i2b={v:k for k,v in self.b2i.items()}
    
if __name__=='__main__':
    # 加载语料
    cn=open('dataset/train-cn.txt','r').read()
    en=open('dataset/train-en.txt','r').read()
    
    # 训练
    tokenizer=BPETokenizer()
    tokenizer.train(text_list=[cn,en],vocab_size=5000)
    
    # 保存
    tokenizer.save('tokenizer.bin')
    
    # 还原
    tokenizer=BPETokenizer()
    tokenizer.load('tokenizer.bin')
    print('vocab size:',tokenizer.vocab_size())
    
    # 编码
    ids,tokens=tokenizer.encode('下雨天和巧克力更配')
    print('encode:',ids,tokens)
    
    # 解码
    s=tokenizer.decode(ids)
    print('decode:',s)