'use strict';
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const crypto=require('node:crypto');
const root=path.resolve(__dirname,'../../../../web/symmetry-lab');
const math=require(path.join(root,'dist/math.js'));
const context={window:{}};vm.runInNewContext(fs.readFileSync(path.join(root,'dist/data.js'),'utf8'),context);
const data=context.window.UGG_DATA;
assert.equal(data.models.length,15);
assert(data.models.every(m=>m.status==='complete'&&m.strict_load));
assert.equal(data.latest_snapshot.total_completed_epochs,2520);
assert.equal(data.latest_snapshot.checkpoint_count,30);
assert.deepEqual(Array.from(data.models,m=>m.name),['mp20_u','c2db51_u','joint_u','mp20_all','c2db51_all','joint_all','mp20_ef','mp20_eg','mp20_eh','c2db51_ef','c2db51_eg','c2db51_eh','joint_ef','joint_eg','joint_eh']);
assert(!('comparison' in data));
assert(data.gallery.every(f=>!f.historical&&typeof f.model_label==='string'));
assert.deepEqual(Array.from(data.sections,s=>s.id),['architecture','datasets','training','generation','construction','pending']);
assert(!data.gallery.some(f=>f.original.includes('model_comparison')||f.original.includes('current_snapshot')||f.original.includes('videos/')));
assert.equal(data.gallery.filter(f=>f.original.includes('14_random_structures')).length,3);
let maxLatticeError=0,maxCoordinateError=0,maxProjectionError=0,cases=0;
function error(a,b){if(Array.isArray(a))return Math.max(0,...a.map((x,i)=>error(x,b[i])));return Math.abs(a-b);}
for(const material of data.materials){
  for(const f of material.fixtures){
    const x=math.expand(material,f.state),p=math.project(material,x,f.candidate),d=math.diagnostics(material,x);
    maxLatticeError=Math.max(maxLatticeError,error(x.L,f.lattice));
    maxCoordinateError=Math.max(maxCoordinateError,error(x.F,f.coordinates));
    maxProjectionError=Math.max(maxProjectionError,error(p.qdot,f.q_velocity));
    assert(d.operation<1e-10);assert(d.metric<1e-10);assert(p.residual<1e-10);cases++;
  }
  for(let i=0;i<21;i++){
    const z=[...material.state];z[0]+=(i-10)*.055;z[z.length-1]=material.kind==='space'?.12+i*.015:.4+i*.12;
    const x=math.expand(material,z),d=math.diagnostics(material,x);assert(d.operation<1e-10);assert(d.metric<1e-10);cases++;
  }
  const x=math.expand(material,material.state),candidate=material.elements.map((_,i)=>i===0?[1,2,3]:[0,0,0]);
  assert(math.project(material,x,candidate).qdot.every(v=>Math.abs(v)<1e-12));
}
assert(maxLatticeError<1e-9);assert(maxCoordinateError<1e-12);assert(maxProjectionError<1e-11);
for(const figure of data.gallery){
 const bytes=fs.readFileSync(path.join(root,'dist',figure.path));assert.equal(crypto.createHash('sha256').update(bytes).digest('hex'),figure.sha256);
}
assert.equal(data.gallery.length,JSON.parse(fs.readFileSync(path.join(root,'gallery_catalog.json'),'utf8')).figures.length);
assert.equal(data.baseline_evaluation.validation.sample_attempts_verified,5404);
assert.equal(data.baseline_evaluation.generation_quality.length,16);
assert.equal(data.gallery.filter(f=>f.status==='待补实验').length,0);
const report={passed:true,python_reference_cases:18,additional_parameter_cases:42,total_cases:cases,max_lattice_error:maxLatticeError,max_coordinate_error:maxCoordinateError,max_projection_error:maxProjectionError,verified_png_files:data.gallery.length,placeholder_figures:0,scope:'JavaScript expansion and physical projection matched to actual PyTorch compiler fixtures; no neural sampling or energy calculation.'};
fs.mkdirSync(path.join(root,'qa'),{recursive:true});fs.writeFileSync(path.join(root,'qa/math_validation.json'),JSON.stringify(report,null,2));console.log(JSON.stringify(report,null,2));

