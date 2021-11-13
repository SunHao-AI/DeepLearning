# 先模拟一张图片输入：

## 数据

> X: 1 * N；W: N * L；Y: 1 * L
>
> 其中：1为输入1张图像；N为图像展开像素，即28*28=784；L为分类个数，即10种类别

> X，MNIST 图像输入数据

$$
\begin{flalign*}

X = 
    \begin{pmatrix}
        x_{1}, & x_{2}, & \cdots ,& x_{n}\\
    \end{pmatrix}_{1*N}
\text{ }&&

\end{flalign*}
$$

> W，权重

$$
\begin{flalign*}

W = 
    \begin{pmatrix}
        w_{11}, & w_{21}, & \cdots ,& w_{nl}\\
        w_{12}, & w_{22}, & \cdots ,& w_{nl}\\
        \vdots & \vdots & \ddots & \vdots\\
        w_{1n}, & w_{2n}, & \cdots ,& w_{nl}\\
    \end{pmatrix}_{N*L}\
\text{ }&&

\end{flalign*}
$$

> Y，one-hot标签

$$
\begin{flalign*}

Y = 
    \begin{pmatrix}
        y_{1}, & y_{2}, & \cdots ,& y_{n}\\
    \end{pmatrix}_{1*N}
\text{ }&&

\end{flalign*}
$$

## 第一步 f(X,W)=X@W +B

$$
\begin{flalign}
f(x,w) = 
	X@W=
	\begin{pmatrix}
        f_{1}, & f_{2}, & \cdots &, f_{l}\\
    \end{pmatrix}_{1*L}

&\text{}&&
\end{flalign}
$$

## 第二步 防止过溢

$$
\begin{flalign}

\begin{split}

f  & = f - f_{max}=>
	\begin{pmatrix}
        f_{1}- f_{max},& f_{2}- f_{max},& \cdots &, f_{n}- f_{max}\\
    \end{pmatrix}_{1*L}&
    \\  
&=>eg:
    \begin{pmatrix}
        0 &, -1.2, & \cdots &, -5\\
    \end{pmatrix}_{1*L}
    
\end{split}&&

\end{flalign}
$$

## 第三步 softmax

$$
\begin{flalign}
code中:
S =	
	\sum_{j=1}^{L}
	\frac{e^{f_{ij}}}{\sum_{k=1}^{L}e^{f_{ik}}}
{\quad\quad}
分子是f一行里的一个元素做指数，分母f一行的元素和

&\text{}&&
\end{flalign}
$$

## 第四步 交叉熵损失

$$
\begin{flalign}
C=-
	\sum_{j=1}^{L}
	(Y_{j}*lnS_{j})

&\text{}&&
\end{flalign}
$$

## 第五步 损失函数

$$
\begin{flalign}
loss = 
	-C.sum()+\frac{\lambda}{2}*\Vert W \Vert_2^2=
	(\sum_{j=1}^{L}(Y_{j}*lnS_{j})).sum()+
	\frac{\lambda}{2}*\Vert W \Vert_2^2


&\text{}&&
\end{flalign}
$$

## 第六步 求导

$$
\begin{flalign}

\begin{split}

& loss = 
	C+\frac{\lambda}{2}*\Vert W \Vert_2^2=
	-Y*lnS +\frac{\lambda}{2}*\Vert x \Vert_2^2
	\\
& C =-Y*lnS \;\;\;\;\;\;\;\; f(x,w) = XW
	\\
& (1)\frac{\partial loss}{\partial w}=
	\frac{\partial C}{\partial w}+\lambda *W
	\\
& (2)\frac{\partial C}{\partial w}=
	\frac{\partial C}{\partial S}*
	{\color{red}\frac{\partial S}{\partial f}}*
	\frac{\partial f}{\partial w}
	\\
& (3)\frac{\partial C}{\partial S}=
	\frac{\partial (-Y*lnS)}{\partial S}=-\frac{Y}{ S}
	\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;
	\frac{\partial f}{\partial W}=\frac{\partial (X@W)}{\partial W}=X^T
	\\
\end{split}&&

\end{flalign}
$$

> ### 核心求导

$$
\begin{flalign}

\begin{split}
& f = 
    \begin{pmatrix}
        f_{1}, & f_{2}, & \cdots ,& f_{l}\\
    \end{pmatrix}_{1*L}
\\
& S = 
	\begin{pmatrix}
        e^{f_1}, & e^{f_2}, & \cdots ,& e^{f_l}
    \\
    \end{pmatrix}_{1*L} =
    \begin{pmatrix}
        S_{1}, & S_{2}, & \cdots ,& S_{l}
    \\
    \end{pmatrix}_{1*L}
\\
&\frac{\partial S}{\partial f}=
	(
		\frac{\partial S_1}{\partial f},
		\frac{\partial S_2}{\partial f},\cdots,
		\frac{\partial S_l}{\partial f}
	)
\\

&=	(
		{\color{blue}
        (\frac{\partial S_1}{\partial f_1}+
        \frac{\partial S_1}{\partial f_2}+\cdots+
        \frac{\partial S_1}{\partial f_l})},
        (\frac{\partial S_2}{\partial f_1}+
        \frac{\partial S_2}{\partial f_2}+\cdots+
        \frac{\partial S_2}{\partial f_l}),
        \cdots,
        (\frac{\partial S_l}{\partial f_1}+
        \frac{\partial S_l}{\partial f_2}+\cdots+
        \frac{\partial S_l}{\partial f_l})
	)
\\
&
令\sum_{k=1}^{L}e^{f_{k}}={e^{f_{1}}+e^{f_{2}}+...+e^{f_{l}}}=h
\\
& 以蓝色部分为例，\frac{\partial S_i}{\partial f_j}可分为2种情况讨论:
\\
& 1)当i=j时，
\frac{\partial S_i}{\partial f_j}=
	\frac{\partial S_i}{\partial f_i}=
	\frac{\partial \frac{e^{f_i}}{e^{f_1}+e^{f_2}+\cdots+e^{f_l}}}{\partial f_i}=
	\frac{\partial \frac{e^{f_i}}{h}}{\partial f_i}=
	\frac{e^{f_i}*h-e^{f_i}*\frac{\partial h}{\partial f_i}}{h^2}
\\
& 其中
\frac{\partial h}{\partial f_i}=
	\frac{\partial (e^{f_1}+e^{f_2}+...+e^{f_L})}{\partial f_i}	=
	e^{f_i},
\\
&则\frac{\partial S_i}{\partial f_i}=
	\frac{e^{f_i}*h-e^{f_i} * \frac{\partial h}{\partial f_i}}{h^2}=
	\frac{e^{f_i}*h-e^{f_i} * e^{f_i}}{h^2}=
	S_i-{S_i}^2
\\
 & 2)当i!=j时，
\frac{\partial S_i}{\partial f_j}=
	\frac{\partial \frac{e^{f_i}}{e^{f_1}+e^{f_2}+\cdots+e^{f_l}}}{\partial f_j}=
	\frac{\partial \frac{e^{f_i}}{h}}{\partial f_j}=
	\frac{-e^{f_i}*\frac{\partial h}{\partial f_j}}{h^2}
\\
& 其中
\frac{\partial h}{\partial f_j}=
	\frac{\partial (e^{f_1}+e^{f_2}+...+e^{f_L})}{\partial f_j}	=
	e^{f_j},
\\
&则\frac{\partial S_i}{\partial f_j}=
	\frac{-e^{f_i}*\frac{\partial h}{\partial f_j}}{h^2}=
	\frac{-e^{f_i}*e^{f_i}}{h^2}=
	-S_i * S_j
\\
& 那么令\frac{\partial S_i}{\partial f}=
	\frac{\partial Si}{\partial f_{1}}+
    \frac{\partial Si}{\partial f_{2}}+...+
    \frac{\partial Si}{\partial f_{l}}=
    \sum_{j=1}^{L}
	\frac{\partial S_i}{\partial f_j}=
	\sum_{j=1}^{L}
	S_j*(\Delta_j-S_i)
\\
& 其中,当i=j时=>\Delta_i=1，当i!=j时=>\Delta_i=0
 
 
 
\end{split}&&

\end{flalign}
$$

> ### 代入公式得

$$
\begin{flalign}

\begin{split}

&\frac{\partial C}{\partial w}=
	\frac{\partial C}{\partial S}*
	{\color{red}\frac{\partial S}{\partial f}}*
	\frac{\partial f}{\partial w}=
	-\frac{Y}{ S}*\sum_{j=1}^{L}
	S_j*(\Delta_j-S_i)
	*X^T 
\\
& =>\frac{\partial C}{\partial w}=
	[
        {\color{green}
        (-\frac{Y}{S}*\sum_{j=1}^{L}
        S_j*(\Delta_j-S_1)
        *X^T )},
        (-\frac{Y}{S}*\sum_{j=1}^{L}
        S_j*(\Delta_j-S_2)
        *X^T ),\cdots,
        (-\frac{Y}{S}*\sum_{j=1}^{L}
        S_j*(\Delta_j-S_l)
        *X^T )
	]
\\
&用绿色部分分析:
-\frac{Y}{S}*\sum_{j=1}^{L}
        S_j*(\Delta_j-S_1)
        *X^T =
        -\sum_{j=1}^{L}
        \frac{Y_j}{S_j}*S_j*(\Delta_j-S_1)*X^T=
        -\sum_{j=1}^{L}
        Y_j*(\Delta_j-S_1)*X^T
\\
&= 
	-(
        \sum_{j=1}^{L} Y_j * \Delta_j 
        - 
        \sum_{j=1}^{L} Y_j * S_1
	)*X^T =
	-(Y_1-S_1)*X^T
\\
& 归纳可得\frac{\partial C}{\partial w}=
    [
		-(Y_1-S_1)*X^T,
		-(Y_2-S_2)*X^T,\cdots,
		-(Y_l-S_l)*X^T
    ]=
\\ &
    -[
		(Y_1-S_1),
		(Y_2-S_2),\cdots,
		(Y_l-S_l)
    ]*X^T=
    -(Y-S)*X^T=>X^T_{N*1} @ (S-Y)_{1*10}=X^T@(S-Y)

\end{split}&&

\end{flalign}
$$

> ### 合并可得

$$
\begin{flalign}

\begin{split}

& \frac{\partial loss}{\partial w}=
	\frac{\partial C}{\partial w}+\lambda *W=
	X^T@(S-Y)+\lambda *W



\end{split}&&

\end{flalign}
$$

# 同理可得多张图片输入

$$
\begin{flalign*}

X = 
    \begin{pmatrix}
        x_{11} & x_{12} & \cdots & x_{1n}\\
        x_{21} & x_{22} & \cdots & x_{2n}\\
        \vdots & \vdots & \ddots & \vdots\\
        x_{m1} & x_{m2} & \cdots & x_{mn}\\
    \end{pmatrix}_{M*N}
\;\;
W=
	\begin{pmatrix}
        w_{11} & w_{21} & \cdots & w_{l1}\\
        w_{12} & w_{22} & \cdots & w_{l2}\\
        \vdots & \vdots & \ddots & \vdots\\
        w_{1n} & w_{2n} & \cdots & w_{ln}\\
    \end{pmatrix}_{N*L}
 
\;\;
Y=
	\begin{pmatrix}
        y_{11} & y_{12} & \cdots & y_{1l}\\
        y_{21} & y_{22} & \cdots & y_{2l}\\
        \vdots & \vdots & \ddots & \vdots\\
        y_{mn} & y_{m2} & \cdots & y_{ml}\\
    \end{pmatrix}_{M*L}
\text{ }&&


\end{flalign*}
$$

> ### 导数公式一样，只是X，W，Y均为矩阵而不是向量了

$$
\begin{flalign}

\begin{split}

& \frac{\partial loss}{\partial w}=
	\frac{1}{M} *
	\frac{\partial C}{\partial w}+\lambda *W=
	\frac{X^T@(S-Y)}{M} +\lambda *W



\end{split}&&

\end{flalign}
$$


