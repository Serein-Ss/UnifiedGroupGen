'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const root=path.resolve(__dirname,'../../../../web/symmetry-lab');
const baseURL=process.env.UGG_TEST_URL||'http://127.0.0.1:8768/';
const {chromium}=require(process.env.UGG_PLAYWRIGHT_MODULE||'playwright');
(async()=>{const browser=await chromium.launch({headless:true,...(process.env.UGG_CHROME_EXECUTABLE?{executablePath:process.env.UGG_CHROME_EXECUTABLE}:{})});try{
const expected=JSON.parse(fs.readFileSync(path.join(root,'qa/theory_build.json'),'utf8'));
const page=await browser.newPage({viewport:{width:1440,height:1050}}),errors=[],failed=[];page.on('pageerror',e=>errors.push(e.message));page.on('requestfailed',r=>failed.push(r.url()));
await page.goto(baseURL,{waitUntil:'networkidle'});
await page.getByRole('link',{name:'数学原理推导',exact:true}).click();await page.waitForLoadState('networkidle');await page.evaluate(()=>document.fonts.ready);
const expectedSections=expected.chapters+1+(expected.h1_guide?1:0)+(expected.h2_guide?1:0)+(expected.h3_guide?1:0);
assert.equal(await page.locator('.math-chapter').count(),expectedSections);assert.equal(await page.locator('.proof-equation').count(),expected.display_equations);assert.equal(await page.locator('.katex-error').count(),0);
if(expected.h1_guide){assert.equal(await page.locator('#appendix-h1').count(),1);assert.equal(await page.locator('#h1-magnetic').count(),1);assert.match(await page.locator('#appendix-h1').innerText(),/整数提升/);}
if(expected.h2_guide){assert.equal(await page.locator('#appendix-h2').count(),1);assert.equal(await page.locator('.h2-system').count(),4);assert.match(await page.locator('#appendix-h2').innerText(),/空矩阵/);}
if(expected.h3_guide){assert.equal(await page.locator('#appendix-h3').count(),1);assert.equal(await page.locator('#h3-mos2').count(),1);assert.match(await page.locator('#appendix-h3').innerText(),/3.223709795/);}
const symbols=await page.locator('.symbol-table tbody tr').count();assert(symbols>=80);
assert.equal(await page.locator('.theory-sidebar nav a').count(),expectedSections);
await page.screenshot({path:path.join(root,'qa/theory_desktop_top.png')});
for(const [id,name] of [['math-10','projection'],['math-14','log_metric'],['math-16','tio2']]){await page.locator('#'+id).scrollIntoViewIfNeeded();await page.screenshot({path:path.join(root,'qa/theory_'+name+'.png')});}
await page.locator('a[href="#symbols"]').first().click();await page.locator('#symbol-query').fill('H0');const n=await page.locator('.symbol-table tbody tr:visible').count();assert(n>0&&n<symbols);
await page.locator('#symbol-query').fill('稳定子');assert((await page.locator('.symbol-table tbody tr:visible').count())>0);
await page.locator('#symbol-query').fill('no-such-symbol-zz');assert(await page.locator('#no-symbols').isVisible());await page.locator('#symbol-query').fill('');
await page.screenshot({path:path.join(root,'qa/theory_symbols.png')});
await page.setViewportSize({width:390,height:844});await page.goto(new URL('theory.html#math-10',baseURL).href,{waitUntil:'networkidle'});await page.evaluate(()=>document.fonts.ready);await page.locator('#math-10').scrollIntoViewIfNeeded();await page.screenshot({path:path.join(root,'qa/theory_mobile_projection.png')});
let size=await page.evaluate(()=>({scroll:document.documentElement.scrollWidth,viewport:innerWidth}));assert(size.scroll<=size.viewport+1,JSON.stringify(size));
await page.goto(new URL('index.html?material=mos2#lab',baseURL).href,{waitUntil:'networkidle'});assert.equal(await page.locator('#parameter-controls input[type=range]').count(),2);size=await page.evaluate(()=>({scroll:document.documentElement.scrollWidth,viewport:innerWidth}));assert(size.scroll<=size.viewport+1);
await page.goto(new URL('index.html#gallery',baseURL).href,{waitUntil:'networkidle'});assert.equal(await page.locator('.gallery-card').count(),await page.evaluate(()=>window.UGG_DATA.gallery.length));
await page.goto('file:///'+path.join(root,'dist/theory.html').replaceAll('\\','/'));await page.evaluate(()=>document.fonts.ready);assert.equal(await page.locator('.proof-equation').count(),expected.display_equations);
assert.equal(await page.evaluate(()=>document.fonts.check('16px KaTeX_Main')),true);
assert.deepEqual(errors,[]);assert.deepEqual(failed,[]);
const result={passed:true,chapters:expected.chapters,h1_guide:!!expected.h1_guide,h2_guide:!!expected.h2_guide,h3_guide:!!expected.h3_guide,glossary_rows:symbols,display_equations:expected.display_equations,formula_errors:0,checks:['third navigation','chapter links','H0 and Chinese symbol search','desktop and mobile layout','cross-link to MoS2','return to current curated gallery','offline formulas and fonts'],console_errors:errors,failed_requests:failed};
fs.writeFileSync(path.join(root,'qa/theory_browser_validation.json'),JSON.stringify(result,null,2));console.log(JSON.stringify(result,null,2));
}finally{await browser.close();}})().catch(e=>{console.error(e.stack);process.exit(1)});
