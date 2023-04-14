import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math



def perfProf(df: pd.DataFrame, costs: list[str]) -> None:
    '''
    Performance profile as defines in [Elizabeth D. Dolan · Jorge J. More ́ Benchmarking Optimization Software with Performance Profiles]
    df : algormeter dataframe
    costs: costs list
    '''
    def calcPerfProf(cost:str) -> dict: 
        data = df[['Problem','Dim','Algorithm','Status',cost]]
        data = data.astype({'Dim':int, cost:float}) 

        problems = data[['Problem','Dim']].drop_duplicates()
        problems['Name'] = problems.apply(lambda row: row.Problem + ' ' +  str(row.Dim), axis=1)
        problems = problems.set_index('Name')
    
        algorithms = df[['Algorithm']].drop_duplicates()
        algorithms = algorithms.Algorithm.values.tolist() # type: ignore 
        algorithms = [(e, e.split('.').pop()) for e in algorithms]

        minmax = data.where(data['Status'] == 'Success').groupby(['Problem','Dim']).aggregate({cost:['min','max']})
        minmax = minmax.astype({cost:float}) 

        def calcRho(prob,dim,status,prCost):
            try:
                cm = minmax.at[(prob,dim, cost),(cost, 'min')]
                if status == 'Success':
                    r = prCost/cm
                else:
                    r = math.inf 
                return r
            except:
                return 0
            
        with pd.option_context('mode.chained_assignment', None):
            data['tau']= data.apply(lambda x : calcRho(x.Problem, x.Dim, x.Status, x[cost]),axis=1)

        numProbs = problems.count().iat[0]
        solvers = {}
        xMax = 0
        for alg, algNick in algorithms:
            dataAlg = data[(data.Algorithm == alg)].sort_values('tau')
            X,Y = [], []
            for _, _, _, _, status, c, tau in dataAlg.itertuples():
                y = data[(data.Algorithm == alg) & (data.tau <= tau)].count().iat[0] / numProbs
                X.append(round(tau,2))
                Y.append(round(y,2))
                if tau < math.inf:
                    xMax = max(xMax,tau) 
            solvers[algNick] = [X,Y]

        xMax *= 1.05
        for solver, xy in solvers.items():
            v=0
            for i, e in enumerate(solvers[solver][0]):
                if e == math.inf:
                    v = solvers[solver][1][i-1] 
                    solvers[solver][0][i] = xMax
                    solvers[solver][1][i] = v 

            solvers[solver][0].append(xMax)
            solvers[solver][1].append(v)

        return solvers
   
    cl = len(costs)
    match cl:
        case 1:
            nr=1;nc=1;figsize = (6,6)
        case 2:
            nr=1;nc=2;figsize=(8, 4)
        case 3 | 4:
            nr=2;nc=2;figsize=(7, 7)
        case _:
            nr=math.ceil(cl/2);nc=2;figsize=(6, 9)

    fig, axs = plt.subplots(nrows=nr, ncols=nc, num='Performance Profiles', figsize=figsize)
    if cl > 1:
        ax = axs.flatten() # type: ignore 
    else:
        ax = np.array([axs])
    fontsize = 8
    for i, c in enumerate(costs):
        solvers = calcPerfProf(c)
        # print(c,'\n', solvers)
        for solver, xy in solvers.items():
            ax[i].step(xy[0], xy[1], where='post', label=solver, alpha=.5)
        ax[i].set_xlabel('τ',fontsize=fontsize)
        ax[i].set_ylabel(r'$P(r_{p,s} \leq \tau: \, 1 \leq s \leq n_s)$',fontsize=fontsize)
        ax[i].set_xscale('log')
        ax[i].set_title(c,fontsize=fontsize)
        ax[i].legend(fontsize=fontsize)
        ax[i].tick_params(axis='both', which='major', labelsize=fontsize)
        ax[i].tick_params(axis='both', which='minor', labelsize=fontsize)

    for i in range(cl,nr*nc):
        ax[i].axis('off')
    fig.tight_layout()


if __name__ == '__main__':

    dfr = pd.read_pickle('perfprof.pickle')
    # dfr = pd.read_csv('aaa.csv')

    # perfProf(dfr, ['f1','Iterations','gf1','Seconds'] + ['f1','Iterations','gf1'])
    # perfProf(dfr, ['f1','Iterations','gf1'] + ['f2','gf2',])
    # perfProf(dfr, ['Seconds'])
    # perfProf(dfr, ['f1'])
    perfProf(dfr, ['f1','Iterations','Seconds','gf1'])

    plt.show(block=True)
