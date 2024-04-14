# multimodal movie box office forecasting

## 文件运行顺序
- step 1
运行`text_encoding.py`文件，对电影的评论和信息介绍进行特征提取，并将提取后的特征存放在``dataset文件夹中  
> python text_encoding.py

- step 2
运行`train.py`文件，模型开始训练  

- step 3
运行`forecast.py`文件，预测  

## 文件介绍
### dataset  
- `pic`:电影宣传海报 `{movie_id}.png`
- `photos`:电影剧照  `{movie_id}_{num}.png`
- `train.xlsx`:训练集电影的信息
- `test.xlsx`:测试集电影的信息test
- `comment.xlsx`:电影评论信息
- `train_comment_text_encoding.pth`:经过运行`text_encoding.py`文件后，生成对训练集的电影的评论的编码
- `test_comment_text_encoding.pth`:经过运行`text_encoding.py`文件后，生成对测试集的电影的评论的编码
- `train_introduction_encoding.pth`:经过运行`text_encoding.py`文件后，生成对训练集的电影的（导演，演员，简介）的编码
- `train_introduction_encoding.pth`:经过运行`text_encoding.py`文件后，生成对测试集的电影的（导演，演员，简介）的编码

### `python`文件
- `text_encoding.py` : 对电影评论和电影信息（导演，演员，简介）进行编码
- `encoder_decoder.py` : 对电影照片，电影上映时间和电影评论进行编码或处理
- `model.py`: 训练模型
- `train.py` : 训练
- `forecast.py` : 预测

