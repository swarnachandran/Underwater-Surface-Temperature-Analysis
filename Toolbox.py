import numpy as np
import pandas as pd
from pandas import Series
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller,kpss,pacf
import statsmodels.api as sm
import seaborn as sns
np.random.seed(6313)
from scipy import signal

def cal_rolling_mean_var(x1,y1): #calculating rolling mean and variance
        rolling_mean=[]
        rolling_variance=[]
        for i in range(1,len(x1)+1):
                    result=np.mean(x1[:i])
                    result_variance=np.var(x1[:i])
                    rolling_mean.append(result)
                    rolling_variance.append(result_variance)

        #print(f"Rolling mean :",rolling_mean)
        #print(f"Rolling variance of :",rolling_variance)
        print("@"*100)

        plt.figure()
        plt.plot(y1, rolling_mean, color='Yellow')
        plt.ylabel("Rolling mean")
        plt.xlabel('Samples')
        plt.title(f"Plot of Rolling mean")
        plt.show()
        plt.figure()
        plt.plot(y1, rolling_variance, color="Purple")
        plt.ylabel(f"Rolling variance")
        plt.xlabel("Samples")
        plt.title(f"Plot of Rolling variance")
        plt.show()

def ADF_cal(x):
       result = adfuller(x)
       print("ADF Statistic: %f" %result[0])
       print('p-value: %f' % result[1])
       print('Critical Values:')
       for key, value in result[4].items():
                   print('\t%s: %.3f' % (key, value))

def kpss_test(z):
       kpsstest = kpss(z, regression='c', nlags="auto")
       kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
       for key,value in kpsstest[3].items():
                   kpss_output['Critical Value (%s)'%key] = value
       print (kpss_output)

#firstorder &secondorder difference
def first_order_difference(l):
       difference=[]
       difference[0]=difference.append([np.nan])
       for j in range(1,len(l)):
                   difference_value=l[j]-l[j-1]
                   difference.append(difference_value)

       return Series(difference)


def second_order_difference(l):
       difference=[]
       difference[0]=difference.append([np.nan])
       difference[1]=difference.append([np.nan])
       for j in range(2,len(l)):
                    difference_value=l[j]-2*l[j-1]+l[j-2]
                    difference.append(difference_value)
       return Series(difference)

def third_order_diff(l):
    difference=[]
    difference[0]=difference.append([np.nan])
    difference[1]=difference.append([np.nan])
    difference[2]=difference.append([np.nan])
    for j in range(3,len(l)+2):
        difference_value = l[j]-l[j-1]
        difference.append((difference_value))
    return Series(difference)

def correlation_coefficient_cal(x,y):
    n=len(x)
    sum_1=0
    sum_2=0
    for i in x:
        sum_1+=i

    mean__x= sum_1/n

    for j in y:
        sum_2+=j

    mean__y =sum_2/n
    difference=0
    denominator_1=0
    denominator_2=0
    for i,j in zip(x,y):
        difference+=(i-mean__x)*(j-mean__y)
        denominator_1+=(i-mean__x)**2
        denominator_2+=(j-mean__y)**2

    d=(denominator_1)**0.5
    d1=(denominator_2)**0.5
    r=difference/(d*d1)
    return r

def ACF(x,lags):
    mean_acf =np.mean(x)
    T=len(x)
    list_1=[]
    prod_1=0
    deno_1=0
    for t in range(0, T):
        deno_1 += (x[t] - mean_acf) ** 2

    for l in range(0,lags+1):
        for t in range(l,T):
            prod_1+=(x[t]-mean_acf)*(x[t-l]-mean_acf)
        acf=float(prod_1/deno_1)
        prod_1 = 0
        list_1.append(acf)
    print("ACF:",list_1)
    return list_1

def estimated_var(x,k):
    sum=0
    for i in range(0,len(x)):
        sum+=i**2
    denominator = 1/(len(x)-k-1)
    variance = np.sqrt((denominator*sum))
    return variance

def movingaverage(data):
    m=int(input("Enter m:"))
    if m%2!=0: #odd
       j = 0
       ma_list =[]
       while (j+m)!=len(data)+1:
           sum = 0
           mean=0
           for i in range(j,j+m):
               sum+=data[i]
           mean=sum/m
           ma_list.append(mean)
           j+=1
           k = int((m - 1) / 2)
       return ma_list,k

    elif m%2==0: #even
        fold=int(input("Folding order:"))
        if fold%2!=0:
            print("Invalid fold value(Fold should be even)")
        else:
            j=0
            ma_list = []
            while (j + m) != len(data) + 1:
                sum = 0
                mean = 0
                for i in range(j, j + m):
                    sum += data[i]
                mean = sum / m
                ma_list.append(mean)
                j += 1
            j=0
            final=[]
            while (j + fold) != len(ma_list) + 1:
                sum = 0
                mean = 0
                for i in range(j, j + fold):
                    sum += ma_list[i]
                mean = sum / fold
                final.append(mean)
                j += 1
            k=int(m/2)
            return final,k

def armaprocess_GPAC():
    T=int(input("Enter number of samples: "))
    mean=int(input("Enter the mean of white noise: "))
    var=int(input("Enter variance of white noise: "))
    na=int(input("Enter AR process order: "))
    nb=int(input("Enter MA process order: "))
    naparam=[0]*na
    nbparam=[0]*nb
    for i in range(0,na):
        naparam[i]=float(input(f"Enter the coefficient of AR:a{i+1}: "))
    for i in range(0,nb):
        nbparam[i]=float(input(f"Enter the coefficient of MA:b{i+1}: "))
    ar=np.r_[1,naparam]
    ma=np.r_[1,nbparam]
    arma_process = sm.tsa.ArmaProcess(ar, ma)
    mean_y = mean* (1 + np.sum(nbparam)) / (1 + np.sum(naparam))
    y = arma_process.generate_sample(T, scale=np.sqrt(var) + mean_y)
    return y

def qvalue_cal(y,an,bn):
    deno=[]
    k=an
    j=bn
    for a in range(k):
        deno.append([])
        for b in range(k):
            deno[a].append(y[np.abs(j+b)])
        j=j-1
    ddeno=round(np.linalg.det(deno),5)
    j=bn
    num=deno[:k-1]
    num.append([])
    for a in range(k):
        num[k-1].append(y[j+a+1])
    dnum=round(np.linalg.det(num),5)
    if ddeno==0:
        return float('inf')
    else:
        qval=dnum/ddeno
        return round(qval,4)

def GPAC(y,k,j):
    q=[]
    for b in range(j):
        q.append([])
        for a in range(1,k+1):
            q[b].append(qvalue_cal(y,a,b))
    gpac=np.array(q).reshape(j,k)
    gpactable=pd.DataFrame(gpac)
    c=np.arange(1,k+1)
    gpactable.columns=c
    print(gpactable)
    sns.heatmap(gpactable,annot=True)
    plt.xlabel('k')
    plt.ylabel('j')
    plt.title('GPAC table')
    plt.show()

    # ACF plot
def acf_plot(lags, acf, samples):
    x = np.arange(0, lags)
    m = 1.96 / np.sqrt(len(samples))
    plt.stem(x, acf, linefmt='r-', markerfmt='bo', basefmt='b-')
    plt.stem(-1 * x, acf, linefmt='r-', markerfmt='bo', basefmt='b-')
    plt.title("ACF")
    plt.axhspan(-m, m, alpha=.2, color='yellow')
    plt.xlabel('Lags')
    plt.ylabel('ACF')
    plt.show()

def acf_pacf_plot_gpac(lags,samples,process):
    x = np.arange(0,lags)
    m = 1.96 / np.sqrt(samples)
    p=pacf(process,lags-1)
    acf=ACF(process,lags)
    fig,((ax1,ax2))=plt.subplots(nrows=2, ncols=1,figsize=(8,8))
    ax1.stem(x,acf, linefmt='r-', markerfmt='bo', basefmt='b-')
    ax1.stem(-1 *x,acf,linefmt='r-',markerfmt='bo', basefmt='b-')
    ax1.set_title("ACF")
    ax1.axhspan(-m,m,alpha=.1,color='yellow')
    ax2.stem(x,p, linefmt='r-', markerfmt='bo', basefmt='b-')
    ax2.stem(-1 * x,p, linefmt='r-', markerfmt='bo', basefmt='b-')
    ax2.set_title("PACF")
    ax2.axhspan(-m, m, alpha=.1, color='yellow')
    fig.supxlabel('Lags')
    fig.supylabel('ACF/PACF')
    fig.tight_layout()
    plt.show()

from statsmodels.graphics.tsaplots import plot_acf , plot_pacf
def ACF_PACF_Plot(y,lags):
 acf = sm.tsa.stattools.acf(y, nlags=lags)
 pacf = sm.tsa.stattools.pacf(y, nlags=lags)

 fig = plt.figure()
 fig.suptitle(f'ACF/PACF of the raw data : {lags}')
 plt.subplot(211)
 plot_acf(y, ax=plt.gca(), lags=lags)
 plt.subplot(212)
 plot_pacf(y, ax=plt.gca(), lags=lags)
 fig.tight_layout(pad=3)
 plt.show()

#LM algorithm
def WN(teta,na,y):
    numerator=[1]+list(teta[na:])
    denominator=[1]+list(teta[:na])
    if len(numerator)!=len(denominator):
        while len(numerator)<len(denominator):
            numerator.append(0)
        while len(denominator)<len(numerator):
            denominator.append(0)
    system=(denominator,numerator,1)
    t,e=signal.dlsim(system,y)
    e=[i[0] for i in e]
    return np.array(e)

def step0(na,nb):
    teta_o=np.zeros(shape=(na+nb,1))
    return teta_o.flatten()

def step1(delta,na,nb,teta,y):
    e_teta=WN(teta,na,y)
    SSE_O=np.dot(e_teta.T,e_teta)
    X=[]
    for i in range(na+nb):
        teta_delta = teta.copy()
        teta_delta[i]=teta[i]+delta
        en=WN(teta_delta,na,y)
        Xi=(e_teta-en)/delta
        X.append(Xi)
    Xfinal=np.transpose(X)
    A=np.dot(Xfinal.T,Xfinal)
    G=np.dot(Xfinal.T,e_teta)
    return A,G,SSE_O

def step2(A,G,mu,na,nb,teta,y):
    n=na+nb
    I=np.identity(n)
    dteta1=A+(mu*I)
    dteta_inv=np.linalg.inv(dteta1)
    delta_teta=np.dot(dteta_inv,G)
    teta_new=teta+delta_teta
    e=WN(teta_new,na,y)
    SSE_new=np.dot(e.T,e)
    if np.isnan(SSE_new):
        SSE_new=10**10
    return SSE_new,delta_teta,teta_new

def step3(max_iter,mu,delta,epsilon,mu_max,na,nb,y):
    num_iter=0
    teta=step0(na,nb)
    SSE=[]
    while num_iter<max_iter:
        A,G,SSE_O=step1(delta,na,nb,teta,y)
        if num_iter == 0:
            SSE.append(SSE_O)
        SSE_new,delta_teta,teta_new=step2(A,G,mu,na,nb,teta,y)
        SSE.append(SSE_new)
        if SSE_new<SSE_O:
            if np.linalg.norm(delta_teta)<epsilon:
                teta_hat=teta_new
                var=SSE_new/(len(y)-A.shape[0])
                A_inv=np.linalg.inv(A)
                cov=var*A_inv
                return SSE,cov,teta_hat,var
            else:
                teta=teta_new
                mu=mu/10
        while SSE_new>=SSE_O:
            mu=mu*10
            if mu>mu_max:
               print('Mu\'s maximum limit is exceeded')
               return None,None,None,None
            SSE_new, delta_teta, teta_new = step2(A, G, mu, na,nb, teta, y)
        num_iter+=1
        teta = teta_new
        if num_iter>max_iter:
            print('Maximum iterations exceeded')
            return None,None,None,None

#Confidence interval
with np.errstate(divide='ignore'):
    np.float64(1.0) / 0.0
def conf_int(cov,params,na,nb):
    print("Confidence Interval:")
    for i in range(na):
        pos=params[i]+2*np.sqrt(cov[i][i])
        neg=params[i]-2*np.sqrt(cov[i][i])
        print(neg,f'<a{i+1}<',pos)
    for i in range(nb):
        pos=params[na+i]+2*np.sqrt(cov[na+i][na+i])
        neg=params[na+i]-2*np.sqrt(cov[na+i][na+i])
        print(neg,f'<b{i+1}<',pos)

#zero-poles cancellation
def zero_poles(params,na):
    y_den=[1]+list(params[:na])
    e_num=[1]+list(params[na:])
    zeros=np.roots(e_num)
    poles=np.roots(y_den)
    print("The roots of numerator are",zeros)
    print("The roots of denominator are",poles)

#Plot of SSE
def plotSSE(SSE):
    iter=np.arange(0,len(SSE))
    plt.plot(iter,SSE,label='SSE')
    plt.xlabel('Number of iterations')
    plt.ylabel('SSE')
    plt.title('SSE vs. #of Iterations')
    plt.legend()
    plt.show()

#Plot one step ahead prediction plot
def onestepplot(y,y_hat):
    plt.plot(y,label='Actual/Train')
    plt.plot(y_hat,label='one step predictions')
    plt.title('Plot of Actual/Train vs. One step prediction')
    plt.legend()
    plt.xlabel('Samples')
    plt.ylabel('Value')
    plt.show()

#Chi-square test
from scipy.stats import chi2
def chi_test(na,nb,lags,Q,e):
    chi_statistic = (Q ** 2).sum().sum()
    dof = (Q.shape[na] - nb) * (Q.shape[na] - nb)
    alpha = 0.01
    chi_critical = chi2.ppf(1 - alpha, dof)
    if chi_statistic > chi_critical:
        print('The residuals are white')
    else:
        print('The residual is not white')

def difference(y,interval):
    diff=[]
    for i in range(interval,len(y)):
        value=y[i]-y[i-interval]
        diff.append(value)
    return diff

def sarima_model():
    T=int(input('Enter number of samples: '))
    mean=eval(input('Enter mean of white nosie: '))
    var=eval(input('Enter variance of white noise: '))
    na = int(input("Enter AR process order: "))
    nb = int(input("Enter MA process order: "))
    naparam = [0] * na
    nbparam = [0] * nb
    for i in range(0, na):
        naparam[i] = float(input(f"Enter the coefficient of AR:a{i + 1}: "))
    for i in range(0, nb):
        nbparam[i] = float(input(f"Enter the coefficient of MA:b{i + 1}: "))
    while len(naparam) < len(nbparam):
        naparam.append(0)
    while len(nbparam) < len(naparam):
        nbparam.append(0)
    ar = np.r_[1, naparam]
    ma = np.r_[1, nbparam]
    e=np.random.normal(mean,np.sqrt(var),T)
    system=(ma,ar,1)
    t,process=signal.dlsim(system,e)
    return process

#Base models
#average
def avg_one(x):
    train=[]
    for i in range(0,len(x)):
        mean=np.mean(x[0:i])
        train.append(mean)
    return train
def avg_hstep(train,test):
    forecast=np.mean(train)
    pred=[]
    for i in range(len(test)):
        pred.append(forecast)
    return pred