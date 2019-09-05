# StuctureOrientedFilter_3D
使用结构导向滤波处理三维地震数据(也可以处理三通道彩色图片)
## 输入/输出
matlab  .mat格式
## 代码环境
armadillo-9.600.6 / matlab C++混合编程
## 数据维度
Inline x Xline x Times (也可以使用其他三维数据，例如彩色图片)
## 注释
* main.cpp : demo，包括读取数据，转换数据格式，处理数据，写数据<br>
* read_mat.h : 包含了读写数据的函数<br>
* StructureOrientedFiltering.h : 结构导向滤波<br>
* pch.h : 预编译头文件<br>
* pch.cpp : 与预编译头对应的源文件<br>
## 效果
#### 处理前：
<img src="https://github.com/yyhhlancelot/StuctureOrientedFilter_3D/blob/master/before.png" width="350"><br>
#### 处理后：
<img src="https://github.com/yyhhlancelot/StuctureOrientedFilter_3D/blob/master/after.png" width="350"><br>
#### 差值：
<img src="https://github.com/yyhhlancelot/StuctureOrientedFilter_3D/blob/master/diff.png" width="350"><br>
