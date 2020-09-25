import numpy as np
import matplotlib.pyplot as plt


class PlotFitResults(object):
    def __init__(self, fitResults, indep_var='x', yerr=None, ax=None, log=None):
        self.x = fitResults.userkws[indep_var]
        self.y = fitResults.data
        
        self.xfit = np.linspace(x.min()*0.9, x.max()*1.1, 51)
        self.yfit = fitResults.model.eval(params=fitResults.params, k=2, I0=1, **{indep_var: xfit})
#         yfit = fitResults.model.eval(params=fitResults.params, **{indep_var: xfit})
        
        if ax is None:
            fig, ax = plt.subplots()
        
        ax.errorbar(x, y, yerr=yerr, fmt='o', color='purple')
        ax.plot(xfit, yfit, color='orange')
        
        if log=='y' or log=='both':
            ax.set_yscale('log')
        if log=='x' or log=='both':
            ax.set_xscale('log')
    

# class PlotContrastFitResults(PlotFitResults):
#     def __init__(self, fitResults, indep_var='kavg', yerr=None, ax=None):
#         super(PlotContrastFitResults, self).__init__(fitResults, indep_var=indep_var, yerr=yerr, ax=ax)
        
#         model = fitResults.model
#         M = 1
#         y = model.eval(self.yfit)