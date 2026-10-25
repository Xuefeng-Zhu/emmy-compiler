async (page) => {
  const check = (value, message) => { if (!value) throw new Error(message); };
  const report = {checks: [], widths: [], requests: []};
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('request', request => report.requests.push(request.url()));
  await page.reload();
  await page.getByRole('heading', {name:'What happens between IR and PTX?'}).waitFor();
  for (let index=1; index<=6; index++) {
    check(await page.locator('#trace-position').textContent() === `${index} / 6`, 'trace position');
    check(await page.locator('#ptx-code .active').count() > 0, 'highlighted real PTX');
    if (index<6) await page.getByRole('button', {name:'Next →'}).click();
  }
  check(await page.getByRole('button', {name:'Next →'}).isDisabled(), 'last step disabled');
  await page.getByRole('button', {name:'← Previous'}).focus();
  await page.keyboard.press('Enter');
  check(await page.locator('#trace-position').textContent() === '5 / 6', 'keyboard trace');
  report.checks.push('all six trace stages and keyboard navigation');
  await page.getByRole('button', {name:'Block 1, thread 2, global index 10, idle, no memory access', exact:true}).click();
  check(await page.locator('#result-value').textContent() === 'No store', 'idle-thread guard');
  await page.getByRole('button', {name:'Block 1, thread 1, global index 9, writes result 9', exact:true}).focus();
  await page.keyboard.press('Space');
  check(await page.locator('#result-value').textContent() === 'result[9] = 28', 'active-thread output');
  report.checks.push('idle/active mapping and keyboard activation');
  const slider=page.getByRole('slider');
  await slider.focus(); await page.keyboard.press('Home');
  check(await slider.inputValue()==='0','slider lower boundary');
  await page.keyboard.press('End'); check(await slider.inputValue()==='4096','slider upper boundary');
  await page.keyboard.press('ArrowLeft'); check(await slider.inputValue()==='4095','slider keyboard increment');
  await page.getByRole('button',{name:'Jump to x = 44.609375'}).click();
  check((await page.locator('#sample-status').textContent()).includes('44.609375'),'witness input');
  check((await page.locator('#sample-values').textContent()).includes('FAILPASS'),'witness decisions');
  await page.getByLabel('Vertical measure').selectOption('budget');
  check(await page.locator('#error-chart .limit').count()===1,'acceptance reference line');
  await page.getByLabel('Vertical measure').selectOption('relative');
  check(await page.locator('#error-chart .limit').count()===0,'relative chart has no misleading constant tolerance');
  const measurements=await page.evaluate(()=>{
    const d=JSON.parse(document.getElementById('emmy-data').textContent);
    return {count:d.exp.rows.length, failures:[2,3].map(j=>d.exp.rows.filter(r=>Math.abs(r[j]-r[1])>d.exp.atol+d.exp.rtol*Math.abs(r[1])).length),
      lines:document.querySelectorAll('#error-chart path').length};
  });
  check(measurements.count===4097 && measurements.failures.join(',')==='251,0' && measurements.lines===2,'embedded measured evidence');
  report.measurements=measurements;report.checks.push('range keyboard boundaries, witness and both measured chart modes');
  for(const width of [1440,768,360,320]){
    await page.setViewportSize({width,height:900});
    await page.waitForTimeout(50);
    const layout=await page.evaluate(()=>({width:innerWidth,scroll:document.documentElement.scrollWidth,
      chartWidth:document.getElementById('error-chart').clientWidth,
      paths:[...document.querySelectorAll('#error-chart path')].every(p=>!p.getAttribute('d').includes('NaN'))}));
    check(layout.scroll<=width,'horizontal overflow at '+width);check(layout.paths,'chart numeric geometry');report.widths.push(layout);
  }
  await page.emulateMedia({reducedMotion:'reduce'});
  check(await page.evaluate(()=>getComputedStyle(document.documentElement).scrollBehavior)==='auto','reduced motion');
  await page.context().setOffline(true);
  await page.getByRole('button',{name:'← Previous'}).click();
  await page.getByLabel('Vertical measure').selectOption('budget');
  check((await page.locator('#chart-note').textContent()).includes('Values above 1 fail'),'offline interaction');
  check(errors.length===0,'browser JavaScript errors: '+errors.join(';'));
  check(report.requests.every(url=>url==='http://127.0.0.1:8769/index.html'),'unexpected network request');
  report.checks.push('reduced-motion preference; offline controls; no subresource requests or JS errors');
  report.userAgent=await page.evaluate(()=>navigator.userAgent);
  await page.context().setOffline(false);
  await page.setViewportSize({width:1440,height:1100});
  return report;
}
