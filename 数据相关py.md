## Internet spider

Several ways to get your page of interest:

1. Simplest way

```python
import requests
r = requests.get('http://www.baidu.com')
print(r.text)
```



2. Binary code

```python
import requests
r = requests.get('http://www.baidu.com')
with open('baidu.png','wb') as fp:
    fp.write(r,content)
```



3. Anti-spider websites

```python
re = quests.get('http://www.zhihu.com')
# re.status_code = 400
# headers可从http测试网站http://httpbin.org或浏览器的“开发者工具”获得
headers = {"User-Agent": "Mozilla/5.0 (X11; Linux i686) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.83 Safari/535.11"}
re = requests.get('http://www.zhihu.com')
# re.status_code = 200
```



---

### 使用BeautifulSoup

```python
>>> from bs4 import BeautifulSoup
>>> markup = '<p class="title"><b>The little Prince</b><p>'
>>> soup = BeautifulSoup(markup,"lxml")
>>> soup.b
<b>The little Prince</b>
>>> soup.p
<p class="title"><b>The little Prince</b></p>
>>> tag = soup.p
>>> tag.string
'The little Prince'
>>> type(tag.string)
<class 'bs4.element.NavigableString'>
>>> soup.findAll('p','title')
[<b>The little Prince</b>]

```

**获取不确定值的内容**

```python
# '<span class="allstar(.*?)main-title-rating" '
import re
>>> pattern_s = re.compile( '<span class="allstar(.*?)main-title-rating" ')
>>> p = re.findall(pattern_s,r.text)

```



**一个简单的抓取前50条左右页面的代码**

```python
import requests, re, time
from bs4 import BeautifulSoup

cnt = 0; i = 0
while cnt <= 50:
    try:
        r = requests.get('https://movie.douban.com/subject/26709258/comments?start={}&limit=20&sort=new_score&status=P'.format(i*20))
    except Exception as err:
        print(err)
        break

    #print(r.status_code)
    soup = BeautifulSoup(r.text,'lxml')
    pattern = soup.findAll('span','short')

    pattern_s = re.compile( '<span class="allstar(.*?)rating" ')
    p = re.findall(pattern_s,r.text)
    print(len(pattern),len(p))

    cnt += len(p)
    i += 1
```



---

### 使用Numpy

```python
import numpy as np
```

*注意！*

*Numpy中的函数很多都是ufunc，在C语言层面实现，因而比其他库中的函数快很多*

*e.g.   np.power()就快于math.pow()*



**创建数组**

```python
>>> np.array([1,2,3])
array([1,2,3])
>>> np.arange(1,5,0.5)
array([1. , 1.5, 2. , 2.5, 3. , 3.5, 4. , 4.5])
>>> np.arange(1,5,dtype=np.float64)
array([1., 2., 3., 4.])
>>> np.random.random((2,2))
array([[0.81997548, 0.43301959],
       [0.26984574, 0.71770428]])
>>> np.linspace(1,2,10,endpoint=False)
array([1. , 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9])

#创建全1的数组
>>> np.ones([2,3])
#创建全0的数组
>>> np.zeros((2,2))
#利用函数返回值创建数组
>>> np.fromfunction(lambda i,j: (i+1)*(j+1),(9,9))
>>> np.fromfunction(lambda x,y,z: (x**2+y**2+z**2)**0.5,(5,5,5))
```



**查看数组属性**

```python
>>> x = np.array([(1,2,3),(4,5,6)])
#查看秩
>>> x.ndim
2
#查看维度
>>> x.shape
(2,3)
```



**数组的操作**

```python
>>> x = np.array([(1,2,3),(4,5,6)])
#维度之间以','分隔
#选择行，从第0行到第1行
>>> x[0:2]
#选择列，第0列和第1列
>>> x[:,[0,1]]

#改变维度创建新数组
>>> y = x.reshape(3,2)
array([[1, 2],
       [3, 4],
       [5, 6]])
#改变原来的维度
>>> x.resize(3,2)

#拼接(vstack垂直，hstack水平)
>>> a1 = np.array([1,3,7])
>>> a2 = np.array([3,5,8])
>>> np.vstack((a1,a2))
array([[1,3,7],
      [3,5,8]])
>>> np.hstack((a1,a2))
array([1,3,7,3,5,8])

#维度相同的数组可以进行相乘与相加操作
>>> a1 * a2
array([ 3, 15, 56])
>>> a1 + a2
array([ 4,  8, 15])
#广播思想，维度不同的数组也可以进行基本运算
>>> x + a1
array([[ 2,  5, 10],
       [ 5,  8, 13]])
```



**数据统计运算**

```python
#全部元素求和
>>> x.sum()
#行求和，注意axis指的是轴，也就是维度。二维数组中行是1维（0），列是2维（1）
>>> x.sum(axis = 0)
array([5,7,9])
#返回最值、索引
>>> x.min()
>>> x.argmin()
#求平均值、方差和标准差
>>> x.mean()
>>> x.var()
>>> x.std()
```



***线性代数应用**

```python
#计算行列式
>>> np.linalg.det(x)
#计算逆矩阵
>>> np.linalg.inv(x)
#计算矩阵内积
>>> np.dot(x,x)
# linalg.solve计算多元一次方程的根
# linalg.eig计算特征值和特征向量
```



---

### 使用Pandas

```python
from pandas import Series
import pandas as pd
import numpy as np

aSer = pd.Series([1,2.0,8+9j], index = ['a','b','c'])
bSer = pd.Series(['apple','peach','lemon'], index = [1,2,3])
print(bSer.index,bSer.values)
print(aSer['b'])
print(np.exp(aSer))

#转换
data = {'lxh':'nb','nz':'cnb','lldq':'fcnb'}
sindex = ['lxh','nz','lldq','dhf']
cSer = pd.Series(data, index = sindex)
print(pd.isnull(cSer))

#数据对齐
dSer = pd.Series({'lxh':'ChongAAA','lldq':'HaoQiang'})
print(cSer + dSer)

#name属性
cSer.name = '电影'
cSer.index.name = '名称'

#方便的求平均等
print(aSer.values.mean())
print(aSer.values.var())
```



**DataFrame**

```python
#DataFrame
data2 = {'name':['MTJJ','AGEN','SHANXIN'],'tg':[8,2,1]}
frame = pd.DataFrame(data2)

data3 = np.array([('MTJJ',8),('AGEN',2),('SHANXIN',1)])
frame = pd.DataFrame(data3, index = range(1,4), columns = ['name','tg'])
print(frame)
print(frame.index)
print(frame.columns)
print(frame.values)

print(frame['name'])
print(frame.tg)
#获取某个区域，一维是行，二维是列
print(frame.iloc[:2,1])

#修改对象属性
frame['name'] = 'admin'
print(frame)
del frame['tg']
print(frame)

#统计功能
frame = pd.DataFrame(data3, index = range(1,4), columns = ['name','tg'])
print(frame.tg.max())
print(frame[frame.tg<='5'])
```

