# AlgorMeter: Tool for developing, testing, measuring and exchange optimizers algorithms

AlgorMeter is a python implementation of an  environment for develop, test, measure, report and  compare optimization algorithms. 
Having a common platform that simplifies developing, testing and exchange of optimization algorithms allows for better collaboration and sharing of resources among researchers in the field. This can lead to more efficient development and testing of new algorithms, as well as faster progress in the field overall.
AlgorMeter produces comparative measures among algorithms  in csv format with effective test function call count.  
It embeds a specific feature devoted to optimize the number of function calls, so that multiple function  calls at the same point are accounted for just once, without storing intermediate results, with benefit in terms of algorithm coding.  
AlgorMeter contains a standard library of 10 DC problems and 7 convex problems for testing algorithms. More problem collections can be easily added.

## problems + algorithms = experiments

- A problem is a function f where f: R(n) -> R with n called dimension.  
- f = f1() - f2() difference of convex function where f1, f2: R(n) -> R. 
- 'problems' is a list of problem
- 'algorithm' is a code that try to find problem local minima
- 'experiment' is an algorMeter run with a list of problems and a list of algorithms that produce a result report

## How to use...

### Implement an algorithm...

Copy and customize algorithm examples like the following *(there are many included example?.py)*

```python
def gradient(p, **kwargs):
    '''Simple gradient'''
    for k in p.loop():
        p.Xkp1 = p.Xk - 1/(k+1) * p.gfXk / np.linalg.norm(p.gfXk) 
```

and refer to the available following system properties

| algorMeter properties | Description
|-----|-----------|
|k, p.K | current iteration |
| p.Xk | current point |
| p.Xkp1 | next point. **to be set for next iteration** |
| p.fXk | p.f(p.Xk) = p.f1(p.Xk) - p.f2(p.Xk)  |
|p.fXkPrev| previous iteration f(x)|
| p.f1Xk | p.f1(p.Xk) |
| p.f2Xk | p.f1(p.Xk) |
| p.gfXk | p.gf(p.Xk) = p.gf1(p.Xk) - p.gf2(p.Xk)  |
| p.gf1Xk | p.gf1(p.Xk) |
| p.gf2Xk | p.gf2(p.Xk) |
| p.optimumPoint | Optimum X |
| p.optimumValue | p.f(p.optimumPoint) |
| p.XStart | Start Point |

to determine the p.Xkp1 for the next iteration.  
...and run it:

```python
df, pv = algorMeter(algorithms = [gradient], problems = probList_covx, iterations = 500, absTol=1E-2)
print('\n', pv,'\n', df)
```

pv and df are pandas dataframe with run result. A .csv file with result is also created in csv folder. 

*(see example\*.py)*

## AlgorMeter interface

```python
def algorMeter(algorithms, problems, tuneParameters = None, iterations = 500, timeout = 180
    runs = 1, trace = False, dbprint= False, csv = True, savedata = False,
     absTol =1.E-4, relTol = 1.E-5,  **kwargs):
```

- algorithms: algorithms list. *(algoList_simple is available )* 
- problems: problem list. See problems list in example4.py for syntax.   *(probList_base, probList_covx, probList_DCJBKM are available)*
- tuneParameters = None: see tuneParameters section 
- iterations = 500: max iterations number 
- timeout = 180: time out in seconds
- runs = 1: see random section 
- trace = False: see trace section 
- dbprint= False: see dbprint section 
- csv = True: write a report in csv format in csv folder
- savedata = False: save data in data folder
- absTol =1.E-4, relTol = 1.E-5: tolerance used in numpy allClose and isClose
- **kwargs: python kwargs propagated to algorithms

call to algorMeter returns two pandas dataframe p1, p2. p2 is a success and fail summary count.
p1 is a detailed report with the following columns.

- Problem  
- Dim
- Algorithm  
- Status: Success, Fail or Error
- Iterations  
- f(XStar  
- f(BKXStar)  
- Delta: absolute difference between  f(XStar) and f(BKXStar)  
- Seconds  
- Start point  
- XStar: minimum
- BKXStar: best known minum
- \f1	f2 gf1	gf2: effective calls count
- ... : other columns with count to counter.up utility (see below)


###  Stop and success condition

```python
    def stop(self) -> bool:
        '''return True if experiment must stop. Override it if needed'''
        if np.array_equal(self.Xk, self.Xprev):
            return False
        return bool(np.isclose(self.fXk,self.fXkPrev,rtol=self.relTol,atol=self.absTol)  
                  or np.allclose (self.gfXk,np.zeros(self.dimension),rtol=self.relTol,atol=self.absTol) )

    def isSuccess(self) -> bool:
        '''return True if experiment success. Override it if needed'''
        return  self.isMinimum(self.XStar)
 
```

can be overriden like in

```python
    def stop():
        return bool(np.isclose(p.f(p.Xk), p.optimumValue,atol=p.absTol, rtol= p.relTol)) or \
                bool(np.allclose (p.Xk, p.optimumPoint,rtol=p.relTol,atol=p.absTol))
    
    p.stop = stop
    p.isSuccess = stop

```

## Problems function call optimization

AlgorMeter embeds a specific feature devoted to optimize the number of function calls, so that multiple function  calls at the same point are accounted for just once, without storing intermediate results, with benefit in terms of algorithm coding.  So in algorithm implementation is not necessary to store the previous result in variables to reduce f1, f2, gf1, gf2 function calls. AlgorMeter cache 128 previous calls to obtain such automatic optimization.  

## Problems ready to use

Importing 'algormeter.libs' probList_base, probList_covx, probList_DCJBKM problems list are available.    
 **probList_DCJBKM** contains ten frequently used unconstrained DC optimization problems, where objective functions are presented as DC (Difference of Convex) functions:
𝑓(𝑥)=𝑓1(𝑥)−𝑓2(𝑥).
 [Joki, Bagirov](https://link.springer.com/article/10.1007/s10898-016-0488-3)

 **probList_covx**  contains  DemMal,Mifflin1, Miffilin2,LQ,QL,MAXQ,MAXL,CB2,CB3,MaxQuad, Rosen, Shor, TR48, A48 and Goffin test convex functions/problem

 **probList_no_covx**  contains special no convex functions: Rosenbrock, Crescent

 **probList_base** contains Parab, ParAbs, Acad simple functions for algorithms early development and test.  

 See 'ProblemsLib.pdf'

### Counters

Instruction like 
> counter.up('lb<0', cls='qp')  

is used to count events in code, summerized in statistics at the end of experiment as a column, available in dataframe returned by call to algorMeter and in final csv.
For the code above a column with count of counter.up calls and head 'qp.lb>0' is produced.  
Also are automatically available columns 'f1', 'f2', 'gf1', 'gf1' with effective calls to f1, f2, gf1, gf2

### dbprint = True

Instruction dbx.print produce print out only if algorMeter call ha option dbptint == True
> dbx.print('t:',t, 'Xprev:',Xprev, 'f(Xprev):',p.f(Xprev) ).  

NB: If dbprint = True python exceptions are not handled and raised.

### Trace == True

If Default.TRACE = True a line with function values are shown as follows in the console for each iteration for algorithms analysis purpose.
>  Acad-2 k:0,f:-0.420,x:[ 0.7 -1.3],gf:[ 1.4 -0.6],f1:2.670,gf1:[ 3.1 -2.9],f2:3.090,gf2:[ 1.7 -2.3]   
 > Acad-2 k:1,f:-1.816,x:[-1.0004 -0.5712],gf:[-8.3661e-04  8.5750e-01],f1:0.419,gf1:[-2.0013 -0.7137],f2:2.235,gf2:[-2.0004 -1.5712]  
> Acad-2 k:2,f:-1.754,x:[-0.9995 -1.4962],gf:[ 9.6832e-04 -9.9250e-01],f1:2.361,gf1:[-1.9985 -3.4887],f2:4.115,gf2:[-1.9995 -2.4962]

These lines represent the path followed by the algorithm for the specific problem.  
NB: If trace = True python exceptions are not handled and raised.

### tuneParameters
Some time is necessary tune some parameter combinations.  Procede as follow (See example4.py):

- Use numeric parameters with Param as domain name, like Param.alpha in your algo code.
- Define a list of lists with possible values of tuning parameters as follows:

```python
tpar = [ # [name, [values list]]
    ('Param.alpha', [1. + i for i in np.arange(.05,.9,.05)]),
    # ('Param.beta', [1. + i for i in np.arange(.05,.9,.05)]),
]
```

- call algorMeter with csv = True and tuneParameters=<list of parameters values> like tuneParameters=tpar.
- open csv file produced and analyze the performance of parameters combinations by looking column '# tunePar'. Useful is a pivot table on such column.

## Random start point 

If algorMeter parameter run is set with a number greater than 1, each algorithm is repeated on the same problem with random start point in range -1 to 1 for all dimensions.
By the method setRandom(center, size) random X can be set in [center-size, center+size] interval.  
See example5.py

## Record data 

with option data == True store in 'npy' folder one file in numpy format, for each experiment with X and Y=f(X) for all iterations.
It is a numpy array with:
> X = data[:,:-1]  
Y = data[:,-1] 

File name is like 'gradient,JB05-50.npy'.  
These files are read by viewer.py data visualizer.

## Minimize

In case you need to find the minimum of a problem/function by applying an algorithm developed with algormeter, the minimize method is available. (See example6.py):

```python
    p = MyProb(K) 
    found, x, y = p.minimize(myAlgo)
```

## Visualizer.py

Running visualizer.py produce or updates contour image in folder 'pics' for each experiment with dimension = 2 with data in folder 'npy'.

# Acknowledgment

Algormeter was inspired and suggested by prof. Manlio Gaudioso of the University of Calabria and made with him.

# Contributing

You can download or fork the repository freely.  
https://github.com/xedla/algormeter  
If you see a mistake you can send me a mail at pietrodalessandro@gmail.com 
If you open up a ticket, please make sure it describes the problem or feature request fully.  
Any suggestion are welcome.
# WARNING
AlgorMeter is still in the early stages of development. 

# License
**If you use AlgorMeter for the preparation of a scientific paper, the citation with a link to this repository would be appreciated.**

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY. 

# Installation
Algormeter is available as pypi pip package.
```python
    pip3 install algormeter
```

# Dependencies
Python version at least
- Python 3.10.6

Package installable with pip3
- numpy
- pandas
- matplotlib

Algormeter plays well with [Visual Studio Code](https://code.visualstudio.com) and in jupyter
