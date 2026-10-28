"""Check the rendered study, controls and numerical agreement.

Run: uv run --with playwright python validate_browser.py
Requires Playwright Chromium (playwright install chromium).
"""
from pathlib import Path
import json
import math
from playwright.sync_api import sync_playwright

OUT=Path(__file__).resolve().parent
checks=[]
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
    for filename in ['explorer.html','retention.html','matched.html','shape.html','tails.html']:
        page=browser.new_page(viewport={'width':900,'height':700},device_scale_factor=1.5)
        errors=[];network=[]
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.on('request',lambda r:network.append(r.url) if r.url.startswith('http') else None)
        page.goto((OUT/filename).as_uri())
        page.wait_for_function('window.studyState !== undefined')
        state=page.evaluate('window.studyState')
        page.screenshot(path=str(OUT/(Path(filename).stem+'_preview.png')),full_page=True)
        values=page.evaluate('''() => {
          const p=studyMath.param(1.3);
          return {sourceTail:2*studyMath.tail(4),sqrtTail:studyMath.tailShare(4,p.sqrt,'sqrt'),
                  flatTail:studyMath.tailShare(4,p.flat,'flat'),
                  sqrtCenter:studyMath.accept(0,p.sqrt,'sqrt'),
                  sqrtMass:2*(studyMath.integral(z=>studyMath.density(z,p.sqrt,'sqrt'),0,p.sqrt.a)+studyMath.integral(z=>studyMath.density(z,p.sqrt,'sqrt'),p.sqrt.a,16)),
                  flatMass:2*(studyMath.integral(z=>studyMath.density(z,p.flat,'flat'),0,p.flat.a)+studyMath.integral(z=>studyMath.density(z,p.flat,'flat'),p.flat.a,16)),
                  sourceCountFromSqrt:studyMath.tailShare(4,p.sqrt,'sqrt')*p.sqrt.r,
                  weightAt5:studyMath.weight(5,2)};
        }''')
        source_tail=math.erfc(4/math.sqrt(2))
        assert math.isclose(values['sourceTail'],source_tail,rel_tol=1e-12)
        assert math.isclose(values['sqrtTail'],source_tail/.24734529870663974,rel_tol=1e-12)
        assert math.isclose(values['flatTail'],source_tail/.44291996643276077,rel_tol=1e-12)
        assert math.isclose(values['sourceCountFromSqrt'],source_tail,rel_tol=1e-12)
        assert math.isclose(values['sqrtCenter'],.18020366547984287,rel_tol=1e-13)
        assert math.isclose(values['sqrtMass'],1,rel_tol=1e-12)
        assert math.isclose(values['flatMass'],1,rel_tol=1e-12)
        assert math.isclose(values['weightAt5'],.5*math.exp(75/8),rel_tol=1e-13)
        for v in ['cap','density','accept','weights','budget','lost','tails','shape']:
            page.select_option('#view',v)
            page.locator('#factor').evaluate("e=>{e.value='1.4';e.dispatchEvent(new Event('input'))}")
            assert math.isclose(page.evaluate('studyState.sqrt.r'),.023303397158667534,rel_tol=1e-12)
            assert math.isclose(page.evaluate('studyState.flat.r/studyState.sqrt.r'),13.474394740083916,rel_tol=1e-8)
            page.locator('#log').uncheck();page.locator('#log').check()
            page.locator('#series input').first.uncheck();page.locator('#series input').first.check()
            page.locator('#hit').hover(position={'x':250,'y':100})
            assert page.locator('#tooltip').is_visible()
            assert 'Expected count' in page.locator('#tooltip').inner_text()
            assert 'Observed n: not applicable' in page.locator('#tooltip').inner_text()
            assert page.locator('#chart path').count()>=2
        for preset in ['shape','cost','matched','limit','double','tails']:
            page.click(f'[data-preset="{preset}"]')
            state_now=page.evaluate('studyState')
            if preset in ['limit','double']:
                assert state_now['sqrt'] is None
                assert ('Limit only' if preset=='limit' else 'Infeasible') in page.locator('#cards').inner_text()
            if preset=='double':assert math.isclose(state_now['flat']['r'],.019965936135145298,rel_tol=1e-13)
        page.fill('#pool','2000000');assert page.evaluate('studyState.sourceN')==2000000
        page.locator('#factor').evaluate("e=>{e.value='1';e.dispatchEvent(new Event('input'))}")
        assert page.evaluate('studyState.sqrt.r')==1
        assert page.evaluate('studyState.flat.r')==1
        page.click('[data-preset="cost"]')
        page.set_viewport_size({'width':740,'height':700})
        height=page.evaluate('document.querySelector("main").getBoundingClientRect().height + 24')
        assert height<=650,(filename,height)
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
        if filename=='retention.html':page.screenshot(path=str(OUT/'retention_740px_preview.png'),full_page=True)
        page.set_viewport_size({'width':400,'height':850})
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
        assert not errors,errors
        assert not network,network
        checks.append({'file':filename,'initial_state':state,'math_checks':values,'height_at_740px':height,'browser_errors':errors,'external_requests':network})
        page.close()
    browser.close()
(OUT/'browser_validation.json').write_text(json.dumps(checks,indent=2))
print(json.dumps(checks,indent=2))
