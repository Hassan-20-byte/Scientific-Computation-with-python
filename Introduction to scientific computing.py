#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


# In[2]:


import IPython


# In[3]:


from IPython.display import Image, display, HTML, Math


# In[4]:


Math(r'\hat{H} = -\frac{1}{2}\epsilon\hat{\sigma}_z-\frac{1}{2}delta\hat{\sigma}_x')


# In[5]:


class QubitHamilton(object):
    def _init_(self, epsilon, delta):
        self.epsilon = epsilon
        self.delta = delta
        
        
    def _rep_latex(self):
        return "$\hat{H}= -%.2f\hat{\sigma}_z-%.2f\hat{\sigma}_x$" %             (self.epsilon/2, self.epsilon/2)


# In[ ]:





# In[6]:


QubitHamilton()


# "$\hat{H}= - .2fhat{\sigma}_z-.2f\hat{\sigma}_x$"

# In[7]:


def f(mu):
    X = stats.norm(loc= mu, scale =np.sqrt(mu))
    N = stats.poisson(mu)
    x = np.linspace (0, X.ppf(0.999))
    n = np.arange(0, x[-1])
    
    
    fig, ax = plt.subplots()
    ax.plot(x, X.pdf(x), color = 'black', lw = 2, label = "Normal($\mu= %d, \ sigma^2= %d$)" % (mu, mu))
    ax.bar(n, N.pmf(n), align = 'edge', label = r"poisson($\lambda= %d$)"  % mu)
    ax.set_ylim(0, X.pdf(x).max() * 1.25)
    ax.legend(loc =2, ncol =2)
    plt.close(fig)
    
    return fig
    


# In[8]:


from ipywidgets import interact
import ipywidgets as widgets


# In[9]:


interact(f, mu=widgets.FloatSlider(min= 1.0, max = 20.0, step =1.0));


# In[10]:


x1 = 5.4 * np.ones(10)
x2 = np.full(10, 5.4)


# In[11]:


x1


# In[12]:


x2


# In[13]:


f = lambda m, n: n + 10 * m
A = np.fromfunction(f, (6, 6), dtype=int)


# In[14]:


A


# In[15]:


A[3:,:3]


# In[16]:


A[::3, ::2]


# In[17]:


A[1::2, 1::3] 


# In[18]:


B = A[1:5, 1:5]
B


# In[19]:


data = np.random.normal(size=(5, 10, 15))
data.sum(axis = 0).shape


# In[20]:


data.sum(axis=(0, 2))


# In[21]:


x = np.linspace(-4, 4, 9)
np.where(x < 0, x**2, x**3)


# In[22]:


x


# In[23]:


np.select([x < -1, x < 2, x >= 2], [x**2 , x**3 , x**4])


# In[2]:


import sympy


# In[25]:


from sympy import I, pi, oo
sympy.init_printing()


# In[26]:


sympy.Rational(11, 13)


# In[27]:


r1 = sympy.Rational(2, 3)
r2 = sympy.Rational(4, 5)


# In[28]:


r1 * r2


# In[29]:


x, y, z = sympy.symbols("x, y, z")


# In[30]:


h = sympy.Lambda(x, x**2)
h


# In[31]:


h(5)


# In[32]:


x = sympy.Symbol("x")
expr =  1 + 2 * x**2 + 3 * x**3
expr


# In[33]:


expr = 2 * (x**2 - x) - x * (x + 1)
expr


# In[34]:


sympy.simplify(expr)


# In[35]:


expr = (x + 1) * (x + 2)
sympy.expand(expr)


# In[36]:


sympy.sin(x + y).expand(trig=True)


# In[37]:


a, b = sympy.symbols("a, b", positive=True)
sympy.log(a * b).expand(log=True)


# In[38]:


sympy.exp(I*a + b).expand(complex=True)


# In[39]:


sympy.expand((a * b)**x, power_base=True)


# In[40]:


sympy.exp((a-b)*x).expand(power_exp=True)


# In[41]:


sympy.factor(x**2 - 1)


# In[42]:


sympy.factor(x * sympy.cos(y) + sympy.sin(z) * x)


# In[43]:


sympy.logcombine(sympy.log(a) - sympy.log(b))


# In[44]:


expr = x + y + x * y * z
expr.collect(x)


# In[45]:


expr = sympy.cos(x + y) + sympy.sin(x - y)
expr.expand(trig=True).collect([sympy.cos(x), sympy.sin(x)]).collect(sympy.cos(x)- sympy.sin(y))


# In[46]:


sympy.apart(1/(x**2 + 3*x + 2), x)


# In[47]:


sympy.together(1 / (y * x + y) + 1 / (1+x))


# In[48]:


sympy.cancel(y / (y * x + y))


# In[49]:


(x + y).subs(x, y)


# In[50]:


sympy.sin(x * sympy.exp(x)).subs(x, y)


# In[51]:


sympy.sin(x * z).subs({z: sympy.exp(y), x: y, sympy.sin: sympy.cos})


# In[52]:


expr = x * y + z**2 *x
values = {x: 1.25, y: 0.4, z: 3.2}
expr.subs(values)


# In[53]:


sympy.N(1 + pi)


# In[54]:


(x + 1/pi).evalf(10)


# In[55]:


expr = sympy.sin(pi * x * sympy.exp(x))
[expr.subs(x, xx).evalf(3) for xx in range(0, 10)]


# In[56]:


expr_func = sympy.lambdify(x, expr)
expr_func(1.0)


# In[57]:


expr_func = sympy.lambdify(x, expr, 'numpy')
import numpy as np
xvalues = np.arange(0, 10)


# In[58]:


expr_func(xvalues)


# ## Calculus Under Sympy

# In[59]:


f = sympy.Function('f')(x)
sympy.diff(f, x) 


# In[60]:


sympy.diff(f, x, x)


# In[61]:


g = sympy.Function('g')(x, y)
sympy.diff(g, x, y)


# In[62]:


g.diff(x, 3, y, 2) 


# In[63]:


expr = x**4 + x**3 + x**2 + x + 1
expr.diff(x)


# In[64]:


expr.diff(x,x)


# In[65]:


expr = (x + 1)**3 * y ** 2 * (z - 1)
expr.diff(x, y, z)


# In[66]:


expr = sympy.sin(x * y) * sympy.cos(x / 2)
expr.diff(x)


# In[69]:


expr = sympy.special.polynomials.hermite(x, 0)
expr.diff(x).doit()


# In[70]:


d = sympy.Derivative(sympy.exp(sympy.cos(x)), x)
d


# In[71]:


d.doit()


# In[72]:


a, b, x, y = sympy.symbols("a, b, x, y")


# In[73]:


f = sympy.Function("f")(x)


# In[74]:


sympy.integrate(f)


# In[75]:


sympy.integrate(f, (x, a, b))


# In[77]:


sympy.integrate(sympy.sin(x))


# In[78]:


sympy.integrate(sympy.sin(x), (x, a, b))


# In[79]:


sympy.integrate(sympy.exp(-x**2), (x, 0, oo))


# In[80]:


sympy.integrate(sympy.sin(x * sympy.cos(x)))


# In[81]:


expr = sympy.sin(x*sympy.exp(y))
sympy.integrate(expr, x)


# In[ ]:





# In[82]:


expr = (x + y)**2
sympy.integrate(expr, x)


# In[83]:


sympy.integrate(expr, x, y)


# In[85]:


sympy.integrate(expr, (x, 0, 1), (y, 0, 1))


# Series Under Sympy

# In[4]:


x, y = sympy.symbols("x, y")
f = sympy.Function("f")(x)


# In[87]:


sympy.series(f, x)


# In[88]:


x0 = sympy.Symbol("{x_0}")


# In[89]:


f.series(x, x0, n=2)


# In[97]:


sympy.cos(x).series()


# In[92]:


sympy.cos(x).series(n = 10)


# In[98]:


sympy.sin(x).series()


# In[100]:


sympy.exp(x).series()


# In[104]:


(1/(1+x)).series()


# In[105]:


expr = sympy.cos(x) / (1 + sympy.sin(x * y))
expr.series(x)


# In[106]:


expr.series(y)


# ## Limit
# 

# In[5]:


sympy.limit(sympy.sin(x) / x, x, 0)


# In[6]:


expr = (x**2 - 3*x) / (2*x - 2)
p = sympy.limit(expr/x, x, sympy.oo)
q = sympy.limit(expr - p*x, x, sympy.oo)
p, q


# In[8]:


n = sympy.symbols("n", integer=True)
x = sympy.Sum(1/(n**2), (n, 1, sympy.oo))
x


# In[9]:


x.doit()


# In[10]:


x = sympy.Product(n, (n, 1, 7))
x


# In[11]:


x.doit()


# In[14]:


x = sympy.Symbol("x")
sympy.Sum((x)**n/(sympy.factorial(n)), (n, 1, sympy.oo)).doit().simplify()


# In[15]:


x = sympy.Symbol("x")


# In[16]:


sympy.solve(x**2 + 2*x - 3)


# In[17]:


a, b, c = sympy.symbols("a, b, c")
sympy.solve(a * x**2 + b * x + c, x)


# In[18]:


sympy.solve(sympy.sin(x) - sympy.cos(x), x)


# In[19]:


sympy.solve(sympy.exp(x) + 2 * x, x)


# In[21]:


eq1 = x + 2 * y - 1
eq2 = x - y + 1


# In[22]:


sympy.solve([eq1, eq2], [x, y], dict=True)


# In[23]:


eq1 = x**2 - y
eq2 = y**2 - x


# In[24]:


sols = sympy.solve([eq1, eq2], [x, y], dict=True)
sols


# In[25]:


sympy.Matrix(3, 4, lambda m, n: 10 * m + n)


# In[ ]:




