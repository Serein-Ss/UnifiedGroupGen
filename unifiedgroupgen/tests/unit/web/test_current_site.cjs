'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),zlib=require('node:zlib');
const root=path.resolve(__dirname,'../../../../web/symmetry-lab'),baseURL=process.env.UGG_TEST_URL||'http://127.0.0.1:8768/';
const {chromium}=require(process.env.UGG_PLAYWRIGHT_MODULE||'playwright');
(async()=>{
 const fixture=JSON.parse(fs.readFileSync(path.join(root,'dist/tables/training_path_example.json'),'utf8'));
 const browser=await chromium.launch({headless:true,...(process.env.UGG_CHROME_EXECUTABLE?{executablePath:process.env.UGG_CHROME_EXECUTABLE}:{})});
 try{
 const page=await browser.newPage({viewport:{width:1440,height:1000}}),errors=[];
 page.on('pageerror',e=>errors.push(e.message));
 await page.goto(baseURL,{waitUntil:'networkidle'});
 assert(await page.locator('#overview-view').isVisible());assert.equal(await page.locator('.topbar [data-view]').count(),4);
 assert.match(await page.locator('#overview-view').innerText(),/MP-20 与 C2DB-51/);
 assert(!/mpts/i.test(await page.locator('body').textContent()));
 await page.screenshot({path:path.join(root,'qa/current_overview.png')});
 await page.locator('.nav[data-view="workflow"]').click();
 assert(await page.locator('#workflow-view').isVisible());assert.equal(await page.locator('[data-workflow-step]').count(),7);
 await page.locator('[data-workflow-step="4"]').click();assert.match(await page.locator('#workflow-state').innerText(),new RegExp('"N": '+fixture.intermediate.decoded.atomic_numbers.length));
 await page.locator('[data-workflow-step="6"]').click();assert.match(await page.locator('#workflow-output').innerText(),/时钟/);
 assert.match(await page.locator('#workflow-state').innerText(),new RegExp('"remaining": '+fixture.supervision.remaining));
 await page.screenshot({path:path.join(root,'qa/current_training.png')});
 await page.locator('[data-phase="sampling"]').click();assert.equal(await page.locator('[data-workflow-step]').count(),8);
 assert.match(await page.locator('#workflow-scope').innerText(),/不代表成功采样/);
 for(let i=0;i<8;i++){await page.locator('[data-workflow-step="'+i+'"]').click();assert.match(await page.locator('#workflow-state').innerText(),/"no_reference_endpoint": true/);}
 await page.screenshot({path:path.join(root,'qa/current_sampling.png')});
 await page.locator('#workflow-prev').click();await page.locator('#workflow-next').click();
 assert.equal(await page.locator('#workflow-next').isEnabled(),false);
 await page.locator('.nav[data-view="lab"]').click();assert.equal(new URL(page.url()).hash,'#lab');assert(await page.locator('#structure').isVisible());
 await page.locator('.nav[data-view="gallery"]').click();assert.equal(await page.locator('.gallery-card').count(),12);
 assert(!/mpts|102,608|102610/i.test(await page.locator('body').innerText()));
 await page.setViewportSize({width:390,height:844});
 for(const name of ['overview','workflow','lab','gallery']){
  await page.locator('.nav[data-view="'+name+'"]').click();
  const size=await page.evaluate(()=>({width:innerWidth,scroll:document.documentElement.scrollWidth}));
  assert(size.scroll<=size.width+1,name+JSON.stringify(size));
 }
 await page.locator('.nav[data-view="workflow"]').click();
 await page.screenshot({path:path.join(root,'qa/current_mobile.png')});
 assert.equal(fixture.record.id,'mp-1064497');assert.equal(fixture.endpoint.decoded.atomic_numbers.length,4);
 assert.equal(fixture.version,'joint_flow_v2');assert.equal(fixture.intermediate.decoded.atomic_numbers.length,4);assert.equal(fixture.supervision.remaining,1);
 for(const forbidden of ['network_output','loss','checkpoint','embedded_nodes'])assert(!(forbidden in fixture),forbidden);
 const manifest=JSON.parse(fs.readFileSync(path.join(root,'dist/data_manifest.json'),'utf8'));
 assert.equal(manifest.records,62134);assert.deepEqual(manifest.datasets,['mp_20','c2db_51']);
 for(const name of ['source_integrity_all_records.csv.gz','dataset_record_statistics.csv.gz']){
  const text=zlib.gunzipSync(fs.readFileSync(path.join(root,'dist/tables',name))).toString('utf8');
  assert(!/mpts/i.test(text));assert.equal(text.trim().split('\n').length-1,62134);
 }
 const result={passed:true,checks:['unified overview and five navigation entries','training and sampling isolated','real TRAIN fixture with no model outputs','old lab bookmark retained','two datasets only in UI, payload and downloads','mobile layout in all views','workflow first/last boundaries'],console_errors:errors};
 assert.deepEqual(errors,[]);fs.writeFileSync(path.join(root,'qa/current_site_validation.json'),JSON.stringify(result,null,2));console.log(JSON.stringify(result));
 }finally{await browser.close();}
})().catch(e=>{console.error(e.stack);process.exit(1);});
