function $(id){return document.getElementById(id)}

function setError(message){
  const el = $('login-error')
  if(!el) return
  if(!message){
    el.classList.add('hidden')
    el.textContent = ''
    return
  }
  el.textContent = message
  el.classList.remove('hidden')
}

function setLoading(isLoading){
  const btn = $('login-btn')
  if(btn) btn.disabled = isLoading
}

async function run(){
  const form = $('login-form')
  if(!form) return

  form.addEventListener('submit', async (e) => {
    e.preventDefault()
    setError('')

    const fd = new FormData(form)
    const payload = {
      username: String(fd.get('username') || '').trim(),
      password: String(fd.get('password') || '').trim(),
    }

    if(!payload.username || !payload.password){
      return setError('Username and password are required.')
    }

    setLoading(true)
    try{
      const res = await fetch('/api/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })

      const data = await res.json().catch(() => null)
      if(!res.ok){
        const msg = data && (data.detail || data.message) ? (data.detail || data.message) : 'Login failed.'
        throw new Error(msg)
      }

      window.location.href = '/valuation'
    }catch(err){
      setError(err && err.message ? err.message : 'Login failed.')
    }finally{
      setLoading(false)
    }
  })
}

run()
