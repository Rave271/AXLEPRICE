function $(id){return document.getElementById(id)}

function safeEl(id){
  const el = $(id)
  return el || null
}

function formatGBP(value){
  try{
    return new Intl.NumberFormat('en-GB', { style: 'currency', currency: 'GBP' }).format(value)
  }catch{
    return `£${Number(value).toFixed(2)}`
  }
}

function toPayload(form){
  const fd = new FormData(form)
  const explain = fd.get('explain') === 'on'
  return {
    brand: String(fd.get('brand') || '').trim(),
    model: String(fd.get('model') || '').trim(),
    year: Number(fd.get('year')),
    mileage: Number(fd.get('mileage')),
    engineSize: Number(fd.get('engineSize')),
    fuelType: String(fd.get('fuelType') || ''),
    transmission: String(fd.get('transmission') || ''),
    repoId: String(fd.get('repoId') || '').trim(),
    hfToken: String(fd.get('hfToken') || ''),
    explain,
  }
}

function setLoading(isLoading){
  const loading = safeEl('loading')
  const submit = safeEl('submit-btn')
  if(loading) loading.classList.toggle('hidden', !isLoading)
  if(submit) submit.disabled = isLoading
}

function setError(message){
  const el = safeEl('error')
  if(!el) return
  if(!message){
    el.classList.add('hidden')
    el.textContent = ''
    return
  }
  el.textContent = message
  el.classList.remove('hidden')
}

function showResult(){
  const empty = safeEl('result-empty')
  const result = safeEl('result')
  if(empty) empty.classList.add('hidden')
  if(result) result.classList.remove('hidden')
}

function renderHistory(items){
  const empty = safeEl('history-empty')
  const list = safeEl('history-list')
  if(!list || !empty) return

  if(!items || !items.length){
    empty.classList.remove('hidden')
    list.classList.add('hidden')
    list.innerHTML = ''
    return
  }

  empty.classList.add('hidden')
  list.classList.remove('hidden')
  list.innerHTML = items.map((item) => {
    const title = `${item.year} ${item.brand} ${item.model}`
    const meta = `${item.mileage.toLocaleString()} miles • ${item.engineSize}L ${item.fuelType} • ${item.transmission}`
    const price = formatGBP(item.predicted_price)
    const conf = `${Number(item.confidence || 0).toFixed(1)}%`
    return `
      <div class="history-item">
        <div>
          <div class="history-title">${title}</div>
          <div class="history-meta">${meta} • ${conf} confidence</div>
        </div>
        <div class="history-price">${price}</div>
      </div>
    `
  }).join('')
}

function setFormValue(form, name, value){
  const el = form && form.elements ? form.elements.namedItem(name) : null
  if(!el) return false
  // namedItem can return RadioNodeList; handle the common single-control case.
  if(typeof el.value !== 'undefined'){
    el.value = String(value)
    return true
  }
  return false
}

function saveDraft(payload){
  try{
    const draft = { ...payload }
    delete draft.hfToken
    localStorage.setItem('valuationDraft', JSON.stringify(draft))
    $('saved-hint').textContent = 'Saved locally'
    setTimeout(() => { $('saved-hint').textContent = '' }, 1200)
  } catch {}
}

function restoreDraft(form){
  try{
    const raw = localStorage.getItem('valuationDraft')
    if(!raw) return
    const d = JSON.parse(raw)
    for(const [k,v] of Object.entries(d)){
      const el = form.elements.namedItem(k)
      if(!el) continue
      if(el.type === 'checkbox') el.checked = Boolean(v)
      else el.value = String(v)
    }
  } catch {}
}

async function run(){
  const form = safeEl('valuation-form')
  if(!form) return
  restoreDraft(form)

  async function loadHistory(){
    try{
      const res = await fetch('/api/history')
      if(res.status === 401){
        window.location.href = '/login'
        return
      }
      const data = await res.json().catch(() => null)
      renderHistory(data && data.items ? data.items : [])
    }catch{}
  }

  await loadHistory()

  const exportBtn = safeEl('export-csv')
  if(exportBtn){
    exportBtn.addEventListener('click', () => {
      window.location.href = '/api/history/export'
    })
  }

  const exampleBtn = safeEl('example-btn')
  if(exampleBtn){
    exampleBtn.addEventListener('click', () => {
      setError('')
      setFormValue(form, 'brand', 'BMW')
      setFormValue(form, 'model', '3 Series')
      setFormValue(form, 'year', '2019')
      setFormValue(form, 'mileage', '30000')
      setFormValue(form, 'engineSize', '2.0')
      setFormValue(form, 'fuelType', 'Petrol')
      setFormValue(form, 'transmission', 'Automatic')
      const brandEl = form.elements.namedItem('brand')
      if(brandEl && brandEl.focus) brandEl.focus()
    })
  }

  const copyBtn = safeEl('copy-btn')
  if(copyBtn) copyBtn.addEventListener('click', async () => {
    const price = (safeEl('predicted-price') || {}).textContent || ''
    const exp = (safeEl('explanation') || {}).textContent || ''
    const text = `Predicted price: ${price}\n\n${exp}`
    try{
      await navigator.clipboard.writeText(text)
      const hint = safeEl('saved-hint')
      if(hint){
        hint.textContent = 'Copied'
        setTimeout(() => { hint.textContent = '' }, 1200)
      }
    }catch{}
  })

  form.addEventListener('submit', async (e) => {
    e.preventDefault()
    setError('')

    const payload = toPayload(form)
    if(!payload.brand) return setError('Brand is required.')
    if(!payload.model) return setError('Model is required.')

    setLoading(true)

    try{
      const res = await fetch('/api/valuate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })

      const data = await res.json().catch(() => null)
      if(!res.ok){
        const msg = data && (data.detail || data.message) ? (data.detail || data.message) : 'Request failed.'
        throw new Error(msg)
      }

      showResult()
      const predicted = safeEl('predicted-price')
      const confidence = safeEl('confidence')
      const summary = safeEl('input-summary')
      if(predicted) predicted.textContent = formatGBP(data.predicted_price)
      if(confidence) confidence.textContent = `${Number(data.confidence || 0).toFixed(1)}%`
      if(summary) summary.textContent = JSON.stringify(data.input, null, 2)

      if(payload.explain){
        const expEl = safeEl('explanation')
        if(expEl){
          expEl.classList.remove('muted')
          expEl.textContent = data.explanation || '(No explanation returned)'
        }
      }else{
        const expEl = safeEl('explanation')
        if(expEl){
          expEl.classList.add('muted')
          expEl.textContent = 'Explanation disabled.'
        }
      }

      await loadHistory()

      saveDraft(payload)
    } catch(err){
      setError(err && err.message ? err.message : 'Something went wrong.')
    } finally {
      setLoading(false)
    }
  })
}

run()
