$$ X = \begin{pmatrix} a_{11} & a_{12} & \cdots & a_{1n}\\ a_{21} & a_{22} & \cdots & a_{2n}\\ \vdots & \vdots & \ddots
& \vdots\\ a_{n1} & a_{n2} & \cdots & a_{nn}\\ \end{pmatrix} = \begin{pmatrix} x_{1} \\ x_{2} \\ \vdots \\ x_{n}\\
\end{pmatrix}\quad\quad 100*784 \quad\quad\quad\quad\quad Y = \begin{pmatrix} y_{11} & y_{12} & \cdots & y_{1n}\\ y_{21}
& y_{22} & \cdots & y_{2n}\\ \vdots & \vdots & \ddots & \vdots\\ y_{n1} & y_{n2} & \cdots & y_{nn}\\ \end{pmatrix} =
\begin{pmatrix} y_{1} \\ y_{2} \\ \vdots \\ y_{n}\\ \end{pmatrix}\quad 100*10 $$

$$ W = \begin{pmatrix} w_{11} & w_{21} & \cdots & w_{n1}\\ w_{12} & w_{22} & \cdots & w_{n2}\\ \vdots & \vdots & \ddots
& \vdots\\ w_{1n} & w_{2n} & \cdots & w_{nn}\\ \end{pmatrix} = \begin{pmatrix} w_{1} &w_{2} & \cdots & w_{n}

    \end{pmatrix}\quad  784*10

$$

### 第一步

$$ f(x,w) = XW= \begin{pmatrix} f_{11} & f_{12} & \cdots & f_{1n}\\ f_{21} & f_{22} & \cdots & f_{2n}\\ \vdots & \vdots
& \ddots & \vdots\\ f_{n1} & f_{n2} & \cdots & f_{nn}\\ \end{pmatrix} = \begin{pmatrix} f_{1} \\ f_{2} \\ \vdots \\ f_
{n}\\ \end{pmatrix}\quad 100*10 $$

### 第二步 防止过溢

$$ f = f - f.max() =>\begin{pmatrix} f_{11}- f.max() & f_{12}- f.max() & \cdots & f_{1n}- f.max()\\ f_{21}- f.max() & f_
{22}- f.max() & \cdots & f_{2n}- f.max()\\ \vdots & \vdots & \ddots & \vdots\\ f_{n1}- f.max() & f_{n2}- f.max() &
\cdots & f_{nn}- f.max()\\ \end{pmatrix}=>\begin{pmatrix} 0 & -1.2 & \cdots & -5\\ -55 & -4 & \cdots & -4\\ \vdots &
\vdots & \ddots & \vdots\\ -5 & -2 & \cdots & -1\\ \end{pmatrix} $$

### 第三步 softmax

$$ { } S=\frac{e^f}{\sum_{i=1}^ne^f} $$

![image-20211108180024540](C:\Users\28654\AppData\Roaming\Typora\typora-user-images\image-20211108180024540.png)

### 第四步 交叉熵损失

$$ C=-Y*lnS $$

### 第五步 损失函数

$$ loss = -Y*lnS +\frac{\lambda}{2}*\Vert W \Vert_2^2 $$

### 第六步 求导

$$ loss = -Y*lnS +\frac{\lambda}{2}*\Vert x \Vert_2^2=C+\frac{\lambda}{2}*\Vert W \Vert_2^2 \newline C=-Y*lnS
\qquad\qquad S=\frac{e^f}{\sum_{i=1}^ne^f} \qquad\qquad f(x,w) = XW \newline \frac{\partial loss}{\partial
W}=\frac{\partial C}{\partial W}+\lambda *W \newline \frac{\partial C}{\partial W}=\frac{\partial C}{\partial S}*
\frac{\partial S}{\partial f}*\frac{\partial f}{\partial W} \newline 其中 \; \frac{\partial C}{\partial
S}=\frac{\partial (-Y*lnS)}{\partial S}=\frac{ -Y}{ S},\qquad\qquad

\frac{\partial f}{\partial W}=\frac{\partial (X*W)}{\partial W}=X^T \newline 假设X=[x_1,x_2,\cdots, x_n]
,Y=[y_1,y_2,\cdots,y_n]^T\qquad\qquad \newline 令\sum_{i=1}^ne^f=h,则\newline \frac{\partial S}{\partial
f}=\frac{\partial (\frac{e^f}{\sum_{i=1}^ne^f})}{\partial f}= \frac{\partial (\frac{e^f}{h})}{\partial f} $$

![image-20211108185345235](C:\Users\28654\AppData\Roaming\Typora\typora-user-images\image-20211108185345235.png)

![image-20211108192343103](C:\Users\28654\AppData\Roaming\Typora\typora-user-images\image-20211108192343103.png)

