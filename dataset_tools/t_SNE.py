import matplotlib.pyplot as plt
import numpy as np
import torch 
from sklearn import datasets
from sklearn import preprocessing
from sklearn.manifold import TSNE
import matplotlib.colors as mcolors

# 数据集导入
clip_proto = torch.load("/root/data1/zbc/erfnet_pytorch/train/erfnet_clip_30.pt")
print(f"-------------------{clip_proto.shape[0:2]}")

embs = []
labels = []

for i in range(clip_proto.shape[0]):
    for j in range(clip_proto.shape[1]):
        embs.append(clip_proto[i,j,:].cpu().numpy().tolist())
        labels.append(i)

embs = np.array(embs)
labels = np.array(labels)
#print(" -------------------y  {}----------------".format(labels.dtype))
#print(" -------------------y  {}----------------".format(type(labels)))
#print(" -------------------y  {}----------------".format(labels.shape))
'''
'''
# t-SNE降维处理

tsne = TSNE(n_components=2, verbose=1 ,random_state=42)
result = tsne.fit_transform(embs)

# 归一化处理
scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
result = scaler.fit_transform(result)


print(" -------------------res  {}----------------".format(result.dtype))
print(" -------------------res  {}----------------".format(type(result)))
print(" -------------------res  {}----------------".format(result.shape))



# 颜色设置
color_ind = [2,7,9,10,11,13,14,16,17,19,20,21,25,28,30,31,32,37,38,40,47,51,
         55,60,65,82,85,88,106,110,115,118,120,125,131,135,139,142,146,147]
color = list(mcolors.CSS4_COLORS.keys())
color = [color[v] for v in color_ind]

# color = ['#FFFAFA', '#BEBEBE', '#000080', '#87CEEB', '#006400',
#         '#00FF00', '#4682B4', '#D02090', '#8B7765', '#B03060']

# 可视化展示
plt.figure(figsize=(10, 10))
plt.title('erfnet encoder prototypes')
plt.xlim((-1.1, 1.1))
plt.ylim((-1.1, 1.1))
for i in range(len(result)):
    plt.text(result[i,0], result[i,1], str(labels[i]), 
             color=color[labels[i]], fontdict={'weight': 'bold', 'size': 9})
for i in range(len(result)):
    plt.scatter(result[i,0], result[i,1], c=color[labels[i]], s=10)    
#plt.scatter(result[:,0], result[:,1], c=labels, s=10)
plt.savefig("./prototypes/erfnet_clip_30.png")


'''
 -------------------x  float64----------------
 -------------------x  <class 'numpy.ndarray'>----------------
 -------------------x  (1797, 64)----------------
 -------------------y  int64----------------
 -------------------y  <class 'numpy.ndarray'>----------------
 -------------------y  (1797,)----------------
'''