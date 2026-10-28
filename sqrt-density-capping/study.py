"""Reproduce the square-root density-capping study locally.

Run: uv run --with numpy --with matplotlib python study.py
No external data, network calls, or empirical samples are used.
"""
from pathlib import Path
import csv
import json
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

OUT=Path(__file__).resolve().parent
SQRT2=math.sqrt(2)
BLUE='#176b9a'; GREEN='#29816c'; ORANGE='#d77536'; GRAY='#77818f'; PURPLE='#7955a7'


def phi(z):
    return math.exp(-z*z/2)/math.sqrt(2*math.pi)


def tail(z):
    return .5*math.erfc(z/SQRT2)


def retained(a, method):
    if method=='flat':return 2*a*phi(a)+2*tail(a)
    return SQRT2*math.exp(-a*a/4)*math.erf(a/2)+2*tail(a)


def factor(a, method):
    r=retained(a,method)
    variance=1+2*a**3*phi(a)/(3*r) if method=='flat' else 2-retained(a,'flat')/r
    return math.sqrt(variance)


def threshold(k, method):
    if k<1:raise ValueError('This thinning family broadens the distribution: k must be >= 1.')
    if method=='sqrt' and k>=SQRT2:return None
    if k==1:return 0.
    lo,hi=0.,max(12.,2*k)
    for _ in range(100):
        a=(lo+hi)/2
        if factor(a,method)<k:lo=a
        else:hi=a
    return (lo+hi)/2


def acceptance(z,a,method):
    divisor=4 if method=='sqrt' else 2
    return math.exp(min(0.,(z*z-a*a)/divisor))


def density(z,a,method,normalized=True):
    r=retained(a,method) if normalized else 1.
    return phi(z)*acceptance(z,a,method)/r


def tail_share(t,a,method):
    r=retained(a,method)
    if t>=a:return 2*tail(t)/r
    inside=2*(a-t)*phi(a) if method=='flat' else SQRT2*math.exp(-a*a/4)*(math.erf(a/2)-math.erf(t/2))
    return (inside+2*tail(a))/r


def crossing_points(k,a,method):
    if k==1:return [0.,a,math.inf]
    beta=.5 if method=='sqrt' else 0.
    r=retained(a,method)
    offset=math.log(k/r)-(1-beta)*a*a/2
    slope=(1/k**2-beta)/2
    points=[0.,a,math.inf]
    if slope!=0 and -offset/slope>0:
        inside=math.sqrt(-offset/slope)
        if inside<a:points.append(inside)
    outer=math.sqrt(2*math.log(k/r)/(1-1/k**2))
    if outer>a:points.append(outer)
    return sorted(set(points))


def tv(k,a,method):
    """Total variation from the same-STD Gaussian, using exact interval masses."""
    if k==1:return 0.
    edges=crossing_points(k,a,method)
    total=0.
    for lo,hi in zip(edges,edges[1:]):
        qmass=tail_share(lo,a,method)-tail_share(hi,a,method)
        gmass=2*(tail(lo/k)-tail(hi/k))
        total+=abs(qmass-gmass)
    return total/2


def record(k,method):
    a=threshold(k,method)
    if a is None:return None
    r=retained(a,method)
    return {'a':a,'r':r,'tv':tv(k,a,method),'center_accept':acceptance(0,a,method)}


def verify():
    nodes,weights=np.polynomial.legendre.leggauss(160)
    def integral(fn,lo,hi):
        return (hi-lo)/2*sum(w*fn((hi+lo)/2+(hi-lo)/2*x) for x,w in zip(nodes,weights))
    checks=[]
    for k in [1,1.1,1.25,1.3,1.4,1.414,SQRT2-1e-5,2]:
        for method in ['sqrt','flat']:
            a=threshold(k,method)
            if a is None:
                checks.append({'k':k,'method':method,'status':'infeasible'});continue
            r=retained(a,method)
            intervals=[(0,a),(a,16)]
            r_num=2*sum(integral(lambda z:density(z,a,method,False),lo,hi) for lo,hi in intervals)
            m_num=2*sum(integral(lambda z:z*z*density(z,a,method,False),lo,hi) for lo,hi in intervals)
            edges=[min(x,16.) for x in crossing_points(k,a,method)]
            tv_num=sum(integral(lambda z:abs(density(z,a,method)-phi(z/k)/k),lo,hi) for lo,hi in zip(edges,edges[1:]))
            assert math.isclose(r_num,r,rel_tol=2e-11),(k,method,r_num,r)
            assert math.isclose(math.sqrt(m_num/r_num),k,rel_tol=2e-11)
            assert math.isclose(tv_num,tv(k,a,method),abs_tol=2e-11)
            checks.append({'k':k,'method':method,'retained':r,'retained_by_quadrature':r_num,
                           'std_by_quadrature':math.sqrt(m_num/r_num),'tv':tv(k,a,method),'tv_by_quadrature':tv_num})
    for method in ['sqrt','flat']:
        grid=[factor(float(a),method) for a in np.linspace(0,8,801)]
        assert all(b>=a-1e-14 for a,b in zip(grid,grid[1:]))
    (OUT/'numerical_validation.json').write_text(json.dumps(checks,indent=2))
    print('Verified retention, STD and total variation by independent split quadrature.')


def make_data():
    selected=[1,1.1,1.2,1.25,1.3,1.35,1.4,1.41,1.414,SQRT2,1.5,2]
    factors=sorted(set([float(x) for x in np.linspace(1,2,201)]+selected+
                       [float(SQRT2-gap) for gap in np.geomspace(1e-7,SQRT2-1.4,81)]))
    records=[{'k':k,'sqrt':record(k,'sqrt'),'flat':record(k,'flat')} for k in factors]
    (OUT/'parameters.json').write_text(json.dumps(records,separators=(',',':')))
    for filename,ks in [('comparison_table.csv',selected),('comparison_curve.csv',factors)]:
        with (OUT/filename).open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['std_factor','method','status','threshold','retained_fraction','discarded_fraction','tv_from_same_std_gaussian','accept_probability_at_mean'])
            for k in ks:
                for method in ['sqrt','flat']:
                    v=record(k,method)
                    w.writerow([k,method,'feasible' if v else ('zero-retention limit only' if k==SQRT2 else 'infeasible'),
                                v['a'] if v else '',v['r'] if v else '',1-v['r'] if v else '',v['tv'] if v else '',v['center_accept'] if v else ''])
    return records


def make_figures(rows):
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    k=1.3;s=record(k,'sqrt');f=record(k,'flat');z=np.linspace(-5,5,2001)
    p=np.exp(-z*z/2)/math.sqrt(2*math.pi)
    rs=np.array([density(float(x),s['a'],'sqrt',False) for x in z])
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    fig.suptitle('Square-root density capping: STD stays below √2 × original',fontsize=18,fontweight='bold')
    ax=axes[0,0]
    ax.fill_between(z,rs,p,color=ORANGE,alpha=.23,label=f"Discarded: {100*(1-s['r']):.2f}%")
    ax.fill_between(z,0,rs,color=BLUE,alpha=.2,label=f"Retained: {100*s['r']:.2f}%")
    ax.plot(z,p,color=GRAY,lw=1.8,label='Original Gaussian')
    ax.plot(z,rs,color=BLUE,lw=2.3,label='Square-root cap')
    ax.set(title='A  Thin to a curved cap (STD factor 1.30)',xlabel='Deviation z (original STD units)',ylabel='Unnormalized density (per unit z)',xlim=(-5,5))
    ax.legend(frameon=False,fontsize=9)
    ax=axes[0,1]
    ax.plot(z,rs/s['r'],color=BLUE,lw=2.3,label='Square-root cap')
    ax.plot(z,[density(float(x),f['a'],'flat') for x in z],color=GREEN,lw=2.3,label='Flat cap')
    ax.plot(z,np.exp(-z*z/(2*k*k))/(k*math.sqrt(2*math.pi)),color=PURPLE,lw=2,ls='--',label='Gaussian target')
    ax.set(title='B  Same mean and STD; different shape (factor 1.30)',xlabel='Deviation z (original STD units)',ylabel='Normalized density (per unit z)',xlim=(-5,5))
    ax.legend(frameon=False,fontsize=9)
    ax=axes[1,0]
    for method,color,label in [('flat',GREEN,'Flat cap'),('sqrt',BLUE,'Square-root cap')]:
        subset=[r for r in rows if r[method]]
        ax.plot([r['k'] for r in subset],[100*r[method]['r'] for r in subset],color=color,label=label,lw=2.4)
    ax.axvspan(SQRT2,2,color=GRAY,alpha=.09)
    ax.axvline(SQRT2,color=BLUE,ls=':',lw=1.5)
    ax.text(1.62,.08,'Square-root cap\nunattainable here',ha='center',color=GRAY)
    ax.scatter([2],[100*record(2,'flat')['r']],color=GREEN,zorder=4)
    ax.annotate('Flat cap at 2×: 1.997% retained',(2,100*record(2,'flat')['r']),xytext=(-12,12),textcoords='offset points',ha='right',color=GREEN,fontsize=9)
    ax.set(yscale='log',xlim=(1,2),ylim=(.00005,150),title='C  Retained data versus STD factor',xlabel='STD factor k (selected / original)',ylabel='Original samples retained (%)')
    ax.legend(frameon=False,fontsize=9,loc='lower left');ax.grid(alpha=.15)
    ax=axes[1,1]
    aa=np.linspace(0,7.5,401)
    for method,color,label in [('flat',GREEN,'Flat cap'),('sqrt',BLUE,'Square-root cap')]:
        ax.plot(aa,[factor(float(a),method) for a in aa],color=color,lw=2.4,label=label)
    ax.axhline(SQRT2,color=BLUE,ls=':',label='√2 STD limit')
    ax.axhline(2,color=GRAY,ls='--',label='2× STD target')
    ax.set(xlim=(0,7.5),ylim=(1,3),title='D  More thinning cannot reach doubled STD',xlabel='Cap half-width a (original STD units)',ylabel='Achieved STD factor k')
    ax.legend(frameon=False,fontsize=9,loc='upper left');ax.grid(alpha=.15)
    for ext in ['png','svg','pdf']:fig.savefig(OUT/('sqrt_capping_overview.'+ext),dpi=180)
    plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(15,4.6),layout='constrained')
    for method,color,label,v in [('sqrt',BLUE,'Square-root cap',s),('flat',GREEN,'Flat cap',f)]:
        axes[0].plot(z,[acceptance(float(x),v['a'],method) for x in z],color=color,lw=2.3,label=label)
        t=np.linspace(0,5,251)
        axes[1].plot(t,[100*tail_share(float(x),v['a'],method) for x in t],color=color,lw=2.3,label=label)
        subset=[r for r in rows if r[method] and r['k']<=SQRT2]
        axes[2].plot([r['k'] for r in subset],[100*r[method]['tv'] for r in subset],color=color,lw=2.3,label=label)
    axes[1].plot(t,[200*tail(float(x)/k) for x in t],color=PURPLE,ls='--',lw=2,label='Gaussian target')
    axes[0].set(title='Acceptance probability (STD factor 1.30)',xlabel='Deviation z (original STD units)',ylabel='Probability of retaining the sample',yscale='log',xlim=(-5,5),ylim=(.03,1.2))
    axes[1].set(title='Tail coverage (STD factor 1.30)',xlabel='Tail threshold t (original STD units)',ylabel='Two-sided tail share (%)',yscale='log',xlim=(0,5))
    axes[2].set(title='Closeness to the Gaussian target',xlabel='STD factor k (selected / original)',ylabel='Total variation from target (%)',xlim=(1,SQRT2))
    for ax in axes:ax.grid(alpha=.15);ax.legend(frameon=False,fontsize=9)
    for ext in ['png','svg','pdf']:fig.savefig(OUT/('sqrt_capping_shape_comparison.'+ext),dpi=180)
    plt.close(fig)


def make_matched_comparison(rows):
    """Compare independently calibrated caps at the same achieved STD."""
    fields=['std_factor','status','flat_retained_fraction','sqrt_retained_fraction',
            'flat_discarded_fraction','sqrt_discarded_fraction',
            'extra_loss_percentage_points','source_pool_multiplier_for_equal_retained_count']
    with (OUT/'matched_std_comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for row in rows:
            s,flat=row['sqrt'],row['flat']
            result={'std_factor':row['k'],'status':'feasible' if s else 'no positive-retention square-root solution',
                    'flat_retained_fraction':flat['r'],'flat_discarded_fraction':1-flat['r']}
            if s:
                result.update(sqrt_retained_fraction=s['r'],sqrt_discarded_fraction=1-s['r'],
                              extra_loss_percentage_points=100*(flat['r']-s['r']),
                              source_pool_multiplier_for_equal_retained_count=flat['r']/s['r'])
            writer.writerow(result)
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for method,color,label in [('sqrt',BLUE,'Square-root cap'),('flat',GREEN,'Flat cap')]:
        valid=[r for r in rows if r[method]]
        axes[0].plot([r['k'] for r in valid],[100*(1-r[method]['r']) for r in valid],color=color,lw=2.5,label=label)
    axes[0].axvspan(SQRT2,2,color=GRAY,alpha=.09)
    axes[0].axvline(SQRT2,color=BLUE,ls=':')
    axes[0].set(xlim=(1,2),ylim=(0,104),xlabel='Achieved STD factor k (selected / original)',
                ylabel='Original data discarded (%)',title='Data loss at the same achieved STD')
    axes[0].text(1.7,25,'Square-root cap\ninfeasible',ha='center',color=GRAY)
    axes[0].legend(frameon=False)
    valid=[r for r in rows if r['sqrt'] and r['k']<=1.414]
    axes[1].plot([r['k'] for r in valid],[r['flat']['r']/r['sqrt']['r'] for r in valid],color=BLUE,lw=2.5)
    r=next(r for r in rows if r['k']==1.4);multiplier=r['flat']['r']/r['sqrt']['r']
    axes[1].scatter([1.4],[multiplier],color=BLUE)
    axes[1].annotate(f'At 1.40× STD: {multiplier:.2f}× source data',(1.4,multiplier),xytext=(-14,-30),textcoords='offset points',ha='right',color=BLUE)
    axes[1].set(yscale='log',xlim=(1,SQRT2),xlabel='Achieved STD factor k (selected / original)',
                ylabel='Source-pool multiplier (square-root / flat)',title='Source data needed for the same retained count')
    axes[1].axvline(SQRT2,color=BLUE,ls=':')
    for ax in axes:ax.grid(alpha=.15)
    for ext in ['png','svg','pdf']:fig.savefig(OUT/('matched_std_comparison.'+ext),dpi=180)
    plt.close(fig)


if __name__=='__main__':
    verify();rows=make_data();make_figures(rows);make_matched_comparison(rows)
    for k in [1.1,1.25,1.3,1.35,1.4,1.41,1.414,2]:
        print(json.dumps({'k':k,'sqrt':record(k,'sqrt'),'flat':record(k,'flat')}))
