"""Build the self-contained interactive figures for the Gaussian sampling report.

Run: uv run --with numpy --with matplotlib python build_sampling_report.py
"""
from pathlib import Path
import csv
import json
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pollarded_gaussian import threshold_for_factor, moments, phi, survival, verify_by_integration

ROOT = Path(__file__).resolve().parent


def cap_tail(u, a, retained):
    return 2 * ((a-u)*phi(a) + survival(a)) / retained if u < a else 2*survival(u)/retained


def build_data():
    factors = sorted(set([1 + i / 200 for i in range(401)] + [math.sqrt(2)]))
    data = []
    for k in factors:
        a = threshold_for_factor(k)
        retained, actual = moments(a)
        data.append({'k': k, 'a': a, 'r': retained, 'c': phi(a), 'tail': survival(a)})
    (ROOT/'sampling_report_data.json').write_text(json.dumps(data, separators=(',', ':')))
    a = threshold_for_factor(2)
    r, _ = moments(a)
    with (ROOT/'sampling_tail_comparison.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['threshold_in_original_std', 'original_tail_share', 'gaussian_target_tail_share', 'density_cap_tail_share'])
        for u in np.linspace(0, 5, 101):
            w.writerow([u, 2*survival(u), 2*survival(u/2), cap_tail(u,a,r)])
    print(json.dumps({'a_for_2':a,'retention_for_2':r,'weight_at_5_for_2':0.5*math.exp(75/8),
                     'tail_beyond_4_gaussian':2*survival(2),'tail_beyond_4_cap':cap_tail(4,a,r),
                     'expected_original_count_beyond_5_per_million':2e6*survival(5)},indent=2))
    return data


HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title><style>
*{box-sizing:border-box}body{margin:0;padding:14px 18px;color:#182230;background:#fff;font:13px/1.4 system-ui,-apple-system,sans-serif}.shell{max-width:1100px;margin:auto}h1{font-size:18px;line-height:1.3;margin:0 0 4px}p{margin:4px 0}.sub,.note{color:#5c6676;font-size:12px}.controls{display:flex;flex-wrap:wrap;gap:8px 16px;align-items:center;margin:12px 0 8px}label{display:inline-flex;gap:7px;align-items:center}input,select,button{font:inherit;accent-color:#146f85}select,input[type=number]{border:1px solid #cbd2da;border-radius:5px;padding:4px;background:white;color:inherit}input[type=number]{width:105px}input[type=range]{width:132px}button{border:1px solid #cbd2da;background:#f5f7f9;color:inherit;border-radius:5px;padding:5px 8px;cursor:pointer}button:hover{background:#e5f0f3}.presets{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:8px}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:8px 0}.card{background:#f3f7f8;border-radius:6px;padding:7px 10px}.card b{display:block;font-size:19px;font-weight:650}.card span{font-size:11px;color:#526174}.series{display:flex;flex-wrap:wrap;gap:7px 14px;font-size:12px;min-height:23px}.series label{cursor:pointer}.swatch{height:3px;width:18px;display:inline-block}.plot{position:relative}svg{width:100%;height:280px;display:block;touch-action:pan-y}.tooltip{display:none;position:fixed;max-width:320px;background:#182230;color:white;padding:10px;border-radius:6px;font-size:11px;line-height:1.6;pointer-events:none;box-shadow:0 3px 14px #0002;z-index:4}.tooltip strong{font-size:12px}.tooltip hr{border:0;border-top:1px solid #ffffff33}.note{min-height:35px}.footer{font-size:10px;color:#667085;border-top:1px solid #eaecf0;padding-top:6px;margin-top:6px}a{color:#146f85}@media(max-width:650px){body{padding:12px}.cards .card b{font-size:16px}.controls{gap:7px}.cards{gap:5px}svg{height:280px}h1{font-size:16px}.sub{font-size:11px}}
</style></head><body><main class="shell"><h1>__TITLE__</h1><p class="sub">One-dimensional Gaussian model · means preserved by symmetry · population calculations</p>
<div class="controls"><label id="viewLabel">View <select id="view"><option value="weights">Gaussian reweighting</option><option value="cap">Density capping</option><option value="density">Normalized distributions</option><option value="budget">Data retained / discarded</option><option value="tails">Tail coverage</option></select></label><label>STD factor k <input id="factor" type="range" min="1" max="3" step="0.01" value="2"><b id="factorValue">2.00</b></label><label>Illustrative source N <input id="pool" type="number" min="1" max="1000000000000" step="1000" value="1000000"></label><label><input id="log" type="checkbox" checked>Log y-axis</label></div>
<div class="presets"><button data-preset="variance">2× variance</button><button data-preset="weights">2× STD: weights</button><button data-preset="budget">2× STD: data cost</button><button data-preset="tails">2× STD: different tails</button></div>
<div class="cards" id="cards"></div><div class="series" id="series"></div><div class="plot"><svg id="chart" viewBox="0 0 900 320" role="img" aria-label="Interactive sampling distribution chart"></svg><div id="tooltip" class="tooltip"></div></div><p class="note" id="caption"></p>
<p class="footer">N is a user-chosen illustration, not an observed dataset. Tooltips report analytical shares and expected counts. No training outcomes or confidence intervals are estimated. Hover the plot for values.</p>
</main><script>
'use strict';
const DATA=__DATA__;
const QUAD=__QUAD__;
const INITIAL='__VIEW__';
const $=id=>document.getElementById(id), COLORS={source:'#667085',target:'#7658b4',cap:'#146f85',lost:'#dd7734',reference:'#b06021'};
let k=2, view=INITIAL, visible={}, chartState=null;
const phi=z=>Math.exp(-z*z/2)/Math.sqrt(2*Math.PI);
const weight=(z,f)=>Math.exp((1-1/(f*f))*z*z/2)/f;
function param(f){return DATA.reduce((a,b)=>Math.abs(b.k-f)<Math.abs(a.k-f)?b:a);}
function number(x,d=5){if(x===0)return '0';if(Math.abs(x)<.0001||Math.abs(x)>=1e7)return x.toExponential(4);return x.toLocaleString('en-US',{maximumFractionDigits:d});}
const pct=x=>number(100*x,6)+'%';
function integrate(fn,l,h,n=48){if(h<=l)return 0;let t=fn(l)+fn(h),step=(h-l)/n;for(let i=1;i<n;i++)t+=(i%2?4:2)*fn(l+i*step);return t*step/3;}
function tail(u){const hi=Math.max(12,u+5),half=(hi-u)/2,mid=(hi+u)/2;return half*QUAD.reduce((sum,[x,w])=>sum+w*phi(mid+half*x),0);}
function capTail(u,p){return u<p.a?2*((p.a-u)*p.c+p.tail)/p.r:2*tail(u)/p.r;}
function mass(fn,l,h,p){let edges=[l,...[-p.a,p.a].filter(x=>x>l&&x<h),h];return edges.slice(1).reduce((s,v,i)=>s+integrate(fn,edges[i],v),0);}
function node(tag,attrs={},text){let n=document.createElementNS('http://www.w3.org/2000/svg',tag);for(let [a,v] of Object.entries(attrs))n.setAttribute(a,v);if(text!==undefined)n.textContent=text;return n;}
function makeSeries(id,name,color,fn,extra={}){return {id,name,color,fn,...extra};}
function config(){const p=param(k), normal=z=>phi(z/k)/k, capped=z=>Math.min(phi(z),p.c), normalized=z=>capped(z)/p.r;
if(view==='weights')return {xmin:-5,xmax:5,xlabel:'Deviation z = (x − μ) / σ (original STD units)',ylabel:'Relative weight (unitless)',series:[makeSeries('target','Gaussian reweighting',COLORS.target,z=>weight(z,k)),makeSeries('reference','2× variance reference',COLORS.reference,z=>weight(z,Math.SQRT2),{dash:true})],caption:'The density ratio gives the Gaussian target in the population. It changes sampling frequency; it cannot add independent observations to a finite tail. Reference: k = √2, which doubles variance.'};
if(view==='cap')return {xmin:-5,xmax:5,xlabel:'Deviation z = (x − μ) / σ (original STD units)',ylabel:'Unnormalized density (per unit z)',series:[makeSeries('source','Original Gaussian',COLORS.source,phi),makeSeries('cap','Retained density before normalization',COLORS.cap,capped)],caption:'Keep each sample with probability min(1, exp((z² − a²)/2)). The retained density is min(φ(z), φ(a)); its area is the retained fraction. Every sample outside ±a is accepted.'};
if(view==='density')return {xmin:-5,xmax:5,xlabel:'Deviation z = (x − μ) / σ (original STD units)',ylabel:'Normalized density (per unit z)',series:[makeSeries('source','Original Gaussian',COLORS.source,phi),makeSeries('target','Gaussian target',COLORS.target,normal),makeSeries('cap','Normalized density cap',COLORS.cap,normalized)],caption:'All curves integrate to one over the full real line. Both broadened distributions have STD factor k and the original mean. Their shapes and tail shares differ.'};
if(view==='budget')return {xmin:1,xmax:3,xlabel:'Target STD factor k = selected STD / original STD',ylabel:'Original samples (%)',percent:true,series:[makeSeries('cap','Data retained',COLORS.cap,x=>param(x).r),makeSeries('lost','Data discarded',COLORS.lost,x=>1-param(x).r)],caption:'The curve is specific to horizontal density capping of a one-dimensional Gaussian. At each factor, retention is calculated from the full distribution, not from a simulated sample.'};
return {xmin:0,xmax:5,xlabel:'Tail threshold t (original STD units): |z| > t',ylabel:'Two-sided tail share (%)',percent:true,series:[makeSeries('source','Original Gaussian',COLORS.source,u=>2*tail(u)),makeSeries('target','Gaussian target',COLORS.target,u=>2*tail(u/k)),makeSeries('cap','Normalized density cap',COLORS.cap,u=>capTail(u,p))],caption:'Each curve gives the two-sided probability beyond threshold t. Equal mean and STD do not imply equal tail coverage. Expected capped counts use N × retained fraction × capped tail share.'};}
function draw(){
$('tooltip').style.display='none';
const p=param(k), N=Math.max(1,Number($('pool').value)||1), cfg=config(), log=$('log').checked;
$('factorValue').textContent=k===Math.SQRT2?'√2 ≈ 1.414':k.toFixed(2);
$('cards').innerHTML='<div class="card"><span>Target STD / variance factors</span><b>'+number(k,4)+'× / '+number(k*k,4)+'×</b></div><div class="card"><span>Cap: original samples retained</span><b>'+pct(p.r)+'</b></div><div class="card"><span>Expected retained from N</span><b>'+number(N*p.r,2)+'</b></div>';
$('caption').textContent=cfg.caption;
$('series').innerHTML='';for(const s of cfg.series){if(visible[s.id]===undefined)visible[s.id]=true;const label=document.createElement('label');label.innerHTML='<input type="checkbox" '+(visible[s.id]?'checked':'')+'><span class="swatch" style="background:'+s.color+'"></span>'+s.name;label.querySelector('input').onchange=e=>{visible[s.id]=e.target.checked;draw();};$('series').append(label);}
const active=cfg.series.filter(s=>visible[s.id]);const W=900,H=320,L=85,R=22,T=16,B=54,pw=W-L-R,ph=H-T-B,steps=300;
let values=[];for(const s of active){s.values=[];for(let i=0;i<=steps;i++){let x=cfg.xmin+(cfg.xmax-cfg.xmin)*i/steps,y=s.fn(x);s.values.push([x,y]);if(y>0&&Number.isFinite(y))values.push(y);}}
let ymin=0,ymax=1;if(values.length){ymax=Math.max(...values)*1.12;if(log){ymin=Math.pow(10,Math.floor(Math.log10(Math.min(...values))));ymax=Math.pow(10,Math.ceil(Math.log10(ymax)));}else if(cfg.percent)ymax=1.05;}
if(log&&cfg.percent)ymax=1;
if(log&&ymin===ymax)ymax=ymin*10;
const xscale=x=>L+(x-cfg.xmin)/(cfg.xmax-cfg.xmin)*pw,yscale=y=>T+ph*(1-(log?(Math.log10(Math.max(y,ymin))-Math.log10(ymin))/(Math.log10(ymax)-Math.log10(ymin)):(y-ymin)/(ymax-ymin)));
const svg=$('chart');svg.replaceChildren();const defs=node('defs'),clip=node('clipPath',{id:'plotClip'});clip.append(node('rect',{x:L,y:T,width:pw,height:ph}));defs.append(clip);svg.append(defs);
let yticks=[];if(log){let first=Math.ceil(Math.log10(ymin)),last=Math.floor(Math.log10(ymax)),stride=Math.max(1,Math.ceil((last-first)/6));for(let n=first;n<=last;n+=stride)yticks.push(10**n);}else for(let i=0;i<=4;i++)yticks.push(ymax*i/4);
for(let y of yticks){let yy=yscale(y);svg.append(node('line',{x1:L,y1:yy,x2:W-R,y2:yy,stroke:'#e7ebef'}));svg.append(node('text',{x:L-10,y:yy+4,'text-anchor':'end','font-size':11,fill:'#536071'},cfg.percent?(y<.00001?(100*y).toExponential(0)+'%':pct(y)):number(y,3)));}
let xticks=view==='budget'?[1,1.25,1.5,1.75,2,2.25,2.5,2.75,3]:view==='tails'?[0,1,2,3,4,5]:[-5,-4,-3,-2,-1,0,1,2,3,4,5];
for(let x of xticks){let xx=xscale(x);svg.append(node('line',{x1:xx,y1:T,x2:xx,y2:T+ph,stroke:'#f0f2f5'}));svg.append(node('text',{x:xx,y:H-B+20,'text-anchor':'middle','font-size':11,fill:'#536071'},number(x,2)));}
svg.append(node('text',{x:L+pw/2,y:H-8,'text-anchor':'middle','font-size':12,fill:'#182230'},cfg.xlabel));svg.append(node('text',{x:19,y:T+ph/2,transform:'rotate(-90 19 '+(T+ph/2)+')','text-anchor':'middle','font-size':12,fill:'#182230'},cfg.ylabel+(log?' · log scale':'')));
for(let s of active){let path='',started=false;for(let [x,y] of s.values){if(!Number.isFinite(y)||(log&&y<=0)){started=false;continue;}path+=(started?'L':'M')+xscale(x).toFixed(2)+','+yscale(y).toFixed(2);started=true;}svg.append(node('path',{d:path,fill:'none',stroke:s.color,'stroke-width':2.6,'stroke-dasharray':s.dash?'6 4':'','clip-path':'url(#plotClip)'}));}
if(view==='budget'){const xx=xscale(k);svg.append(node('line',{x1:xx,y1:T,x2:xx,y2:T+ph,stroke:COLORS.cap,'stroke-dasharray':'3 4'}));}
const cursor=node('line',{id:'cursor',x1:0,y1:T,x2:0,y2:T+ph,stroke:'#182230','stroke-dasharray':'3 4',visibility:'hidden'});svg.append(cursor);
svg.append(node('rect',{id:'hit',x:L,y:T,width:pw,height:ph,fill:'transparent'}));
chartState={cfg,active,p,N,L,pw,xscale,yscale};$('hit').onpointermove=hover;$('hit').onpointerleave=()=>{$('tooltip').style.display='none';$('cursor').setAttribute('visibility','hidden');};
window.reportState={k,view,retained:p.r,threshold:p.a,sourceN:N};
}
function hover(e){const {cfg,active,p,N,L,pw,xscale}=chartState,svg=$('chart'),rect=svg.getBoundingClientRect();const sx=(e.clientX-rect.left)/rect.width*900;let x=cfg.xmin+(Math.min(L+pw,Math.max(L,sx))-L)/pw*(cfg.xmax-cfg.xmin);x=view==='budget'?param(x).k:Math.round(x*100)/100;
$('cursor').setAttribute('x1',xscale(x));$('cursor').setAttribute('x2',xscale(x));$('cursor').setAttribute('visibility','visible');
let rows=['<strong>'+(view==='budget'?'STD factor k':view==='tails'?'Tail threshold t':'Deviation z')+' = '+number(x,4)+'</strong>'];
for(const s of active){let y=s.fn(x),share,count,detail;
if(view==='budget'){share=y;count=N*share;detail='of original data';}
else if(view==='tails'){share=y;count=N*share*(s.id==='cap'?p.r:1);detail=s.id==='cap'?'of retained data; count from N original':s.id==='target'?'of Gaussian target; count if N fresh target draws':'of original data';}
else if(view==='weights'){share=mass(phi,x-.05,x+.05,p);count=N*share;detail='original data in z ± 0.05';}
else{share=mass(s.fn,x-.05,x+.05,p);count=N*share*(view==='density'&&s.id==='cap'?p.r:1);detail=view==='density'&&s.id==='target'?'target bin; count if N fresh target draws':view==='density'&&s.id==='cap'?'retained bin; count from N original':'original pool bin (z ± 0.05)';}
rows.push('<hr><b style="color:'+s.color+';filter:brightness(1.7)">'+s.name+'</b><br>Value: '+(cfg.percent?pct(y):number(y,8))+'<br>Share: '+pct(share)+' · '+detail+'<br>Expected count: '+number(count,4));}
rows.push('<hr>Observed n: not applicable (analytical model)');
const tip=$('tooltip');tip.innerHTML=rows.join('');tip.style.display='block';const parent=$('chart').parentElement.getBoundingClientRect();let left=e.clientX-parent.left+14;if(left+tip.offsetWidth>parent.width)left=e.clientX-parent.left-tip.offsetWidth-14;tip.style.left=Math.max(8,Math.min(e.clientX+14,window.innerWidth-tip.offsetWidth-8))+'px';tip.style.top=Math.max(8,Math.min(e.clientY-40,window.innerHeight-tip.offsetHeight-8))+'px';
}
function setPreset(p){k=p==='variance'?Math.SQRT2:2;view=p==='variance'?'weights':p;$('factor').value=k;$('view').value=view;$('log').checked=['weights','budget','tails'].includes(view);visible={};draw();}
$('view').value=view;$('log').checked=['weights','budget','tails'].includes(view);
$('view').onchange=e=>{view=e.target.value;visible={};$('log').checked=['weights','budget','tails'].includes(view);draw();};
$('factor').oninput=e=>{k=Number(e.target.value);draw();};$('pool').oninput=draw;$('log').onchange=draw;
document.querySelectorAll('[data-preset]').forEach(b=>b.onclick=()=>setPreset(b.dataset.preset));
window.samplingMath={phi,weight,param,tail,capTail,integrate};draw();
</script></body></html>'''


def make_tail_static():
    a=threshold_for_factor(2);r,_=moments(a)
    t=np.linspace(0,5,401)
    fig,ax=plt.subplots(figsize=(8,4.6),layout='constrained')
    for label,color,y in [('Original Gaussian','#667085',[2*survival(x) for x in t]),
                           ('Gaussian target: STD ×2','#7658b4',[2*survival(x/2) for x in t]),
                           ('Density cap: STD ×2','#146f85',[cap_tail(x,a,r) for x in t])]:
        ax.plot(t,100*np.array(y),color=color,label=label,lw=2.4)
    ax.set(yscale='log',xlabel='Tail threshold t (original STD units)',ylabel='Two-sided tail share P(|z| > t) (%)',title='Equal STD does not imply equal tail coverage',xlim=(0,5))
    ax.grid(alpha=.2);ax.legend(frameon=False)
    fig.savefig(ROOT/'sampling_tail_comparison.png',dpi=180)
    plt.close(fig)


def make_weight_statics():
    z=np.linspace(-5,5,1001)
    for filename,k,log in [('gaussian_variance_weights',math.sqrt(2),False),
                           ('gaussian_variance_weights_log',math.sqrt(2),True),
                           ('gaussian_doubled_std_weights_log',2,True)]:
        fig,ax=plt.subplots(figsize=(9,5.2),layout='constrained')
        w=np.exp((1-1/k**2)*z*z/2)/k
        ax.plot(z,w,color='#2563eb',lw=2.6)
        if log: ax.set_yscale('log')
        ax.set(xlabel=r'Deviation in original standard deviations, $z=(x-\mu)/\sigma$',
               ylabel='Relative weight'+(' (log scale)' if log else ''),xlim=(-5,5),
               title='Gaussian reweighting: '+('doubled variance' if k<2 else 'doubled STD'))
        ax.set_xticks(range(-5,6));ax.grid(alpha=.2)
        for point in [0,3,5]:
            value=math.exp((1-1/k**2)*point**2/2)/k
            ax.scatter([point],[value],color='#2563eb',s=30)
            ax.annotate(f'{value:,.2f}',(point,value),xytext=(-10,9) if point==5 else (8,9),
                        textcoords='offset points',ha='right' if point==5 else 'left')
        for ext in ['png','svg']:fig.savefig(ROOT/(filename+'.'+ext),dpi=180)
        plt.close(fig)


def make_icon():
    svg='''<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256" viewBox="0 0 256 256"><rect width="256" height="256" rx="56" fill="#e9f4f2"/><path d="M123 225v-81M128 188l-54-43M128 179l57-35" stroke="#725142" stroke-width="16" stroke-linecap="round"/><path d="M35 148Q23 123 50 108H205Q230 123 218 148Q201 176 176 160Q151 182 127 164Q104 182 78 162Q49 177 35 148Z" fill="#146f85"/><path d="M73 84q17-26 37-18q15-20 36-6q21-9 38 24" fill="none" stroke="#df8b43" stroke-width="12" stroke-linecap="round" stroke-dasharray="3 20"/><path d="M44 44l64 58M46 108l60-62" stroke="#41576a" stroke-width="9" stroke-linecap="round"/><circle cx="40" cy="40" r="13" fill="none" stroke="#df8b43" stroke-width="9"/><circle cx="40" cy="112" r="13" fill="none" stroke="#df8b43" stroke-width="9"/><circle cx="76" cy="76" r="6" fill="#f9fbfc"/></svg>'''
    (ROOT/'sampling_report_icon.svg').write_text(svg)


if __name__=='__main__':
    verify_by_integration()
    data=build_data()
    for filename,title,view in [
        ('sampling_results.html','Broadening a Gaussian: weights, shape and data cost','density'),
        ('sampling_weights.html','Gaussian reweighting concentrates weight in rare tails','weights'),
        ('sampling_retention.html','Density capping trades retained data for action spread','budget'),
        ('sampling_tails.html','Equal STD can hide large differences in tail coverage','tails')]:
        nodes,weights=np.polynomial.legendre.leggauss(64)
        quad=json.dumps(list(zip(nodes.tolist(),weights.tolist())),separators=(',',':'))
        (ROOT/filename).write_text(HTML.replace('__TITLE__',title).replace('__VIEW__',view).replace('__DATA__',json.dumps(data,separators=(',',':'))).replace('__QUAD__',quad))
    make_tail_static()
    make_weight_statics()
    make_icon()
