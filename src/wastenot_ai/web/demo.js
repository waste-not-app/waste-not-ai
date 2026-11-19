'use strict';
const $ = id => document.getElementById(id);
const nonce = document.querySelector('meta[name="wastenot-demo"]').content;
let config, stream = null, photoBlob = null, photoURL = null, detection = null, selected = null;
let manualMode = false, cameraVersion = 0, flowVersion = 0, recVersion = 0;
let detectController = null, recController = null, detectRequestId = null, recRequestId = null;
let photoExpiry = null, catalogFilter = 'all', detecting = false, recommending = false;
const messages = {
  AI_BUSY: 'The model is still finishing the previous photo. Wait a moment, then try again.',
  AI_TIMEOUT: 'That took too long. Your photo and answers are still here, so you can try again.',
  AI_UNAVAILABLE: 'The model isn\'t ready. Check the demo terminal and try again.',
  FILE_TOO_LARGE: 'Images can be at most 5 MiB. Choose a smaller photo.',
  UNSUPPORTED_MEDIA_TYPE: 'Use a JPEG or PNG image.', INVALID_IMAGE: 'We couldn\'t read that image.',
  IMAGE_DIMENSIONS_EXCEEDED: 'That image is too large. Use at most 4096 pixels per side and 12 megapixels.',
  DEMO_SESSION_REQUIRED: 'The demo session ended. Reload the page to start again.',
  DETECTION_NOT_FOUND: 'That detection result is gone. Take or send the photo again.',
  AI_SESSION_EXPIRED: 'The photo session ended. Start a new scan.',
  DEMO_HOURLY_LIMIT: 'The demo hit its Gemini call limit. The method still comes from the guide rules.',
  DEMO_SESSION_LIMIT: 'The demo has too many sessions. Clear old results or restart the server.',
  INVALID_CONTEXT: 'Check the item type, material, and condition you chose.',
  OBJECT_CONFIRMATION_REQUIRED: 'Confirm the object and its condition first.',
  OBJECT_REQUIRES_MANUAL: 'Pick this object manually from the catalog.',
  IDEMPOTENCY_CONFLICT: 'The request changed. Clear it and send the photo again.',
  LLM_NOT_CONFIGURED: 'Gemini isn\'t configured, so the method comes from the guide rules.',
  LLM_MODEL_NOT_CONFIGURED: 'Set a GEMINI_MODEL your account can use, then restart the demo.',
  LLM_UNAVAILABLE: 'We can\'t reach Gemini. Check your connection, the model, and your account\'s API access.',
  LLM_RATE_LIMITED: 'Gemini is rate limited right now, so the method comes from the guide rules.',
  LLM_TIMEOUT: 'Gemini timed out, so the method comes from the guide rules.',
  LLM_BLOCKED_OR_INCOMPLETE: 'Gemini\'s response was blocked or incomplete, so we didn\'t use it.',
  NO_REVIEWED_GUIDE: 'There\'s no reviewed guide for this case yet. To try draft guides, run the demo with --demo-guides.',
  NO_ALLOWED_METHOD: 'The guide has no automatic method for this condition. Fill in what you know, or choose how to handle it yourself.',
  INVALID_MODEL_OUTPUT: 'The model\'s suggestion failed validation, so we didn\'t use it.',
  INVALID_EVIDENCE: 'No matching guide evidence supports that suggestion. Choose how to handle it yourself.',
  RISK_DETAILS_REQUIRED: 'Tell us what kind of risk you checked. Ordinary drink residue isn\'t the same as hazardous material.',
  CONTEXT_REQUIRED: 'Fill in the missing condition, dirt type, or whether it washes off. We won\'t use Reduce in place of an answer.',
  RISK_NOT_COVERED: 'Follow the specialist handling advice below while you confirm a suitable handler.'
};
const rawNames = {'bottle':'Bottle','cup':'Cup or mug','bowl':'Bowl','wine glass':'Wine glass',
  'aluminium_can':'Aluminium can','glass_bottle':'Glass bottle','hdpe_bottle':'HDPE bottle · confirm contents','plastic_bottle':'Plastic bottle · confirm PET',
  'fork':'Fork','spoon':'Spoon','knife':'Knife','vase':'Vase','person':'Person','dining table':'Dining table',
  'cell phone':'Mobile phone','book':'Book','pizza':'Pizza','chair':'Chair','laptop':'Laptop','tv':'TV or monitor'};
function rawLabel(raw) {
  const broad = {'can':'Can · confirm material','plastic_bag':'Plastic bag',
    'paper':'Paper','cardboard':'Cardboard','drink_carton':'Drink carton',
    'plastic_wrapper':'Plastic wrap or film','food_container':'Food container · confirm material',
    'bottle_cap':'Bottle cap · confirm material'};
  if (broad[raw]) return broad[raw];
  return rawNames[raw] || config?.catalog.items.find(x => x.baseline_raw_class === raw)?.label || raw;
}
function message(text, kind = '') {
  $('global-message').textContent = text;
  $('global-message').className = `message ${kind}`;
  $('global-message').hidden = !text;
}
function setStep(number) {
  for (let i = 1; i <= 4; i++) {
    if (i === number) $(`step-${i}`).setAttribute('aria-current', 'step');
    else $(`step-${i}`).removeAttribute('aria-current');
  }
}
const apiPrefix = (window.location.pathname.endsWith('/') ? window.location.pathname : window.location.pathname + '/') + 'demo/api/';
async function api(path, options = {}) {
  let response;
  try {
    response = await fetch(`${apiPrefix}${path}`, {...options,
      headers: {'X-WasteNot-Demo': nonce, ...options.headers}});
  } catch (error) {
    if (error.name === 'AbortError') throw error;
    throw new Error('Can\'t reach the demo server. Check the terminal and restart it if needed.');
  }
  if (response.status === 204) return null;
  const data = await response.json();
  if (!response.ok) {
    const error = new Error(messages[data.detail] || 'That didn\'t work. Check your answers or start a new scan.');
    if (data.data?.status === 'UNAVAILABLE' && Array.isArray(data.data.steps)) error.recommendation = data.data;
    throw error;
  }
  return data;
}
function stopCamera() {
  cameraVersion++;
  if (stream) stream.getTracks().forEach(track => track.stop());
  stream = null;
  $('camera').srcObject = null;
  $('camera').hidden = true;
  $('live-label').hidden = true;
  $('capture').hidden = true;
  $('capture').disabled = true;
  $('start-camera').hidden = false;
  $('camera-status').textContent = photoBlob ? 'Photo ready to review' : 'Camera off';
}
async function startCamera() {
  stopCamera();
  const version = cameraVersion;
  if (!navigator.mediaDevices?.getUserMedia) {
    message('The browser camera isn\'t available. Open the demo at http://127.0.0.1 or use Choose photo.', 'warning');
    return;
  }
  $('camera-status').textContent = 'Waiting for camera permission…';
  try {
    const next = await navigator.mediaDevices.getUserMedia({audio: false,
      video: {facingMode: {ideal: 'environment'}, width: {ideal: 1280}, height: {ideal: 720}}});
    if (version !== cameraVersion) { next.getTracks().forEach(track => track.stop()); return; }
    stream = next;
    $('camera').srcObject = stream;
    $('camera').hidden = false;
    await $('camera').play();
    if (version !== cameraVersion) return;
    $('photo-stage').hidden = true;
    $('camera-placeholder').hidden = true;
    $('live-label').hidden = false;
    $('start-camera').hidden = true;
    $('capture').hidden = false;
    $('capture').disabled = false;
    $('camera-status').textContent = 'Camera on';
    message('Nothing goes to YOLO until you take a photo and press Send for detection.');
  } catch (error) {
    if (version !== cameraVersion) return;
    stopCamera();
    $('camera-placeholder').hidden = !!photoBlob;
    $('photo-stage').hidden = !photoBlob;
    const text = error.name === 'NotAllowedError' ? 'Camera permission wasn\'t granted. Allow the camera in your browser, or choose a photo from your device.' :
      error.name === 'NotFoundError' ? 'No camera found. Use Choose photo instead.' :
      'We couldn\'t open the camera. Close other apps using it, then try again or choose a photo.';
    message(text, 'warning');
  }
}
function clearRecommendation() {
  recVersion++;
  recController?.abort();
  recController = null;
  recommending = false;
  recRequestId = null;
  $('recommendation-result').hidden = true;
  $('recommendation-empty').hidden = false;
  $('recommendation-empty').querySelector('p').textContent = 'Once you confirm the object and its condition, the advice shows up here.';
  $('recommend').innerHTML = 'Get advice <span aria-hidden="true">↗</span>';
  updateForm();
}
function resetConditions() {
  ['cleanliness','integrity','reusable','cleanable','soiling','risk-type','former-contents'].forEach(id => { $(id).value = 'UNKNOWN'; });
  $('reuse-intention').value = '';
  $('recommendation-goal').value = 'AUTO';
  $('recycling-access').value = 'UNKNOWN';
  $('confirmed').checked = false;
  $('risk').checked = false;
  updateConditionDetails();
  $('former-contents-field').hidden = true;
}
function updateConditionDetails() {
  const dirty = $('cleanliness').value === 'DIRTY';
  $('cleaning-details').hidden = !dirty;
  if (!dirty) { $('cleanable').value = 'UNKNOWN'; $('soiling').value = 'UNKNOWN'; }
  $('risk-details').hidden = !$('risk').checked;
  if (!$('risk').checked) $('risk-type').value = 'UNKNOWN';
}
function updateForm() {
  $('recommend').disabled = !config || !$('object-key').value || !$('confirmed').checked || recommending || detecting;
}
function clearAnalysis() {
  flowVersion++;
  detectController?.abort();
  detectController = null;
  detection = null; selected = null; manualMode = false; detecting = false; detectRequestId = null;
  $('boxes').replaceChildren(); $('object-list').replaceChildren(); $('detection-section').hidden = true;
  $('detect').textContent = 'Send for detection'; $('detect').disabled = false;
  $('selected-label').textContent = 'Nothing selected';
  $('correct-detection').hidden = true;
  resetConditions(); fillObjectOptions([]); clearRecommendation(); setStep(1);
}
async function resetServer() {
  if (config) await api('session', {method: 'DELETE'});
}
async function clearFlow() {
  stopCamera(); clearAnalysis();
  clearTimeout(photoExpiry);
  if (photoURL) URL.revokeObjectURL(photoURL);
  photoURL = null; photoBlob = null;
  $('photo').removeAttribute('src'); $('photo-stage').hidden = true; $('camera-placeholder').hidden = false;
  $('detect').hidden = true; $('retake').hidden = true; $('file').value = '';
  $('camera-status').textContent = 'Camera off';
  message('Photo and results cleared. You can start again.');
  try { await resetServer(); } catch (error) { message(error.message, 'warning'); }
}
async function preparePhoto(blob) {
  stopCamera(); clearAnalysis();
  const version = flowVersion;
  try { await resetServer(); } catch (error) { message(error.message, 'error'); return; }
  if (version !== flowVersion) return;
  if (photoURL) URL.revokeObjectURL(photoURL);
  photoBlob = blob; photoURL = URL.createObjectURL(blob);
  $('photo').src = photoURL; $('photo-stage').hidden = false; $('camera-placeholder').hidden = true;
  $('detect').hidden = false; $('retake').hidden = false; $('start-camera').hidden = true;
  $('camera-status').textContent = 'Review photo';
  message('Check the photo, then press Send for detection when you\'re ready.');
  clearTimeout(photoExpiry);
  photoExpiry = setTimeout(() => { clearFlow(); message('We cleared the photo after 15 minutes. Start a new scan if you need to.'); }, 15 * 60 * 1000);
}
function canvasBlob(canvas) { return new Promise((resolve,reject) => canvas.toBlob(blob => blob ? resolve(blob) : reject(new Error('Couldn\'t capture the photo.')), 'image/jpeg', .9)); }
async function capture() {
  if (!$('camera').videoWidth) { message('Wait for the camera preview to load.', 'warning'); return; }
  const version = ++flowVersion;
  const canvas = document.createElement('canvas');
  const ratio = Math.min(1,1600 / Math.max($('camera').videoWidth,$('camera').videoHeight));
  canvas.width = Math.round($('camera').videoWidth * ratio); canvas.height = Math.round($('camera').videoHeight * ratio);
  canvas.getContext('2d').drawImage($('camera'),0,0,canvas.width,canvas.height);
  stopCamera();
  try { const blob = await canvasBlob(canvas); if (version === flowVersion) await preparePhoto(blob); } catch (error) { message(error.message,'error'); }
  canvas.width = 0; canvas.height = 0;
}
async function filePhoto(file) {
  if (!file) return;
  if (!['image/jpeg','image/png'].includes(file.type) || file.size > 5 * 1024 * 1024) {
    message('Choose a JPEG or PNG image up to 5 MiB.','error'); return;
  }
  const version = ++flowVersion;
  let bitmap;
  try {
    bitmap = await createImageBitmap(file);
    if (bitmap.width * bitmap.height > 12000000 || Math.max(bitmap.width,bitmap.height) > 4096) throw new Error(messages.IMAGE_DIMENSIONS_EXCEEDED);
    const scale = Math.min(1,1600 / Math.max(bitmap.width,bitmap.height));
    const canvas = document.createElement('canvas'); canvas.width = Math.round(bitmap.width*scale); canvas.height = Math.round(bitmap.height*scale);
    canvas.getContext('2d').drawImage(bitmap,0,0,canvas.width,canvas.height);
    const blob = await canvasBlob(canvas);
    if (version === flowVersion) await preparePhoto(blob);
    canvas.width = 0; canvas.height = 0;
  } catch (error) { message(error.message || 'We couldn\'t read that photo.','error'); }
  finally { bitmap?.close(); }
}
async function detectPhoto() {
  if (!photoBlob || detecting) return;
  detectRequestId ||= crypto.randomUUID();
  const version = flowVersion;
  detecting = true; updateForm(); clearRecommendation();
  $('detect').disabled = true; $('detect').textContent = 'YOLO is looking…';
  detectController = new AbortController();
  message('Running detection on the photo. The camera never streams video.');
  try {
    const result = await api('detect',{method:'POST',body:photoBlob,signal:detectController.signal,
      headers:{'Content-Type':'image/jpeg','X-Client-Request-Id':detectRequestId}});
    if (version !== flowVersion) return;
    detection = result.data; selected = null; manualMode = false;
    renderDetections(); setStep(2);
    const count = detection.objects.filter(x => x.supported_for_review).length;
    message(count ? 'Pick one object on the photo or from the list below, then confirm its type and condition.' :
      'We didn\'t find anything we can suggest. Retake the photo or pick the item from the catalog.',count ? '' : 'warning');
  } catch (error) { if (version === flowVersion && error.name !== 'AbortError') message(error.message,'error'); }
  finally {
    if (version === flowVersion) { detecting = false; $('detect').disabled = false; $('detect').textContent = 'Detect again'; updateForm(); }
  }
}
function renderDetections() {
  $('boxes').replaceChildren(); $('object-list').replaceChildren(); $('detection-section').hidden = false;
  $('detection-summary').textContent = `${detection.objects.length} candidates`;
  for (const object of detection.objects) {
    const label = rawLabel(object.raw_class), score = Math.round(object.confidence*100);
    const button = document.createElement('button');
    button.type = 'button'; button.className = `bbox ${object.supported_for_review ? '' : 'unsupported'} ${selected?.object_id === object.object_id ? 'selected' : ''}`;
    button.disabled = !object.supported_for_review;
    const [x1,y1,x2,y2] = object.bbox;
    Object.assign(button.style,{left:`${x1*100}%`,top:`${y1*100}%`,width:`${(x2-x1)*100}%`,height:`${(y2-y1)*100}%`,
      zIndex:String(object.supported_for_review ? Math.round((1-(x2-x1)*(y2-y1))*1000)+1 : 0),
      pointerEvents:object.supported_for_review ? 'auto' : 'none'});
    const span = document.createElement('span'); span.textContent = `Looks like: ${label} · ${score}%`; button.append(span);
    button.setAttribute('aria-label',`Pick ${label}, detection score ${score}%`);
    button.addEventListener('click',() => selectObject(object)); $('boxes').append(button);
    const item = document.createElement('button'); item.type = 'button'; item.className = `object-choice ${selected?.object_id === object.object_id ? 'selected' : ''}`;
    item.disabled = !object.supported_for_review; item.setAttribute('aria-pressed',String(selected?.object_id === object.object_id));
    const title = document.createElement('strong'); title.textContent = `Looks like: ${label}`;
    const detail = document.createElement('span'); detail.textContent = `${score}% · ${object.ambiguous_with?.length ? 'Overlapping classes · check' : object.confidence_band === 'UNCERTAIN' ? 'Needs review' : object.supported_for_review ? 'Pick object' : 'Pick manually'}`;
    item.append(title,detail); item.addEventListener('click',() => selectObject(object)); $('object-list').append(item);
  }
}
function fillObjectOptions(suggested) {
  const select = $('object-key'); select.replaceChildren(new Option('Confirm the type or material',''));
  if (!config) return;
  if (suggested.length) {
    const group = document.createElement('optgroup'); group.label = 'Matches the detected object';
    config.catalog.items.filter(x => suggested.includes(x.object_key)).forEach(x => group.append(new Option(x.label,x.object_key)));
    select.append(group);
  }
  const others = document.createElement('optgroup'); others.label = suggested.length ? 'Correct to another object' : 'Item catalog';
  config.catalog.items.filter(x => !suggested.includes(x.object_key)).forEach(x => others.append(new Option(x.label,x.object_key)));
  select.append(others);
}
function selectObject(object) {
  selected = object; manualMode = false; resetConditions(); clearRecommendation();
  $('correct-detection').hidden = false;
  fillObjectOptions(object.suggested_object_keys || []);
  $('selected-label').textContent = 'Looks like: ' + rawLabel(object.raw_class);
  $('context-help').textContent = object.ambiguous_with?.length ? 'The model made several guesses for the same area. Compare them in the object list, then confirm the right material.' : object.confidence_band === 'UNCERTAIN' ? 'This guess needs review. A high score doesn\'t prove the type or material.' : 'Object selected. Confirm its type, material, and condition.';
  renderDetections(); setStep(3); updateForm();
}
function chooseManual(key = '') {
  stopCamera();
  flowVersion++;
  detectController?.abort(); detecting = false;
  $('detect').disabled = false; $('detect').textContent = 'Send for detection';
  $('photo-stage').hidden = !photoBlob; $('camera-placeholder').hidden = !!photoBlob;
  selected = null; manualMode = true; resetConditions(); clearRecommendation(); fillObjectOptions([]);
  $('correct-detection').hidden = true;
  $('object-key').value = key; $('selected-label').textContent = 'Picked manually';
  $('context-help').textContent = 'You picked this from the catalog, so it isn\'t treated as a YOLO detection.';
  if (detection) renderDetections(); setStep(3); updateMaterialNote(); updateForm();
  $('context-title').scrollIntoView({behavior:'smooth',block:'center'});
}
function updateMaterialNote() {
  const item = config?.catalog.items.find(x => x.object_key === $('object-key').value);
  $('former-contents-field').hidden = !(item?.object_key === 'hdpe_bottle' || item?.guide_template === 'broad_food_container');
  $('material-note').textContent = item?.guide_template === 'general_handling' ? 'Use this for items that aren\'t in the catalog. Check the material and risks. You\'ll get starter advice even if we can\'t pick a final method.' :
    item?.special_handling ? 'Follow the warnings and the specialist handler route for this item.' :
    item?.object_key === 'hdpe_bottle' ? 'Confirm it\'s HDPE (code 2) and what it held. The resin code doesn\'t mean it\'s safe to reuse. This guide only covers food and drink bottles.' :
    item?.guide_template === 'broad_food_container' ? 'Confirm what it held. This guide only covers food and drink packaging. Chemical packaging needs a specialist handler.' :
    item?.guide_template === 'food' ? 'Reduce advice helps you avoid leftovers next time. A photo can\'t tell whether food is safe to eat.' :
    item?.guide_template === 'toothbrush' ? 'Confirm it\'s a manual plastic toothbrush with no battery or electrics. Electric toothbrushes follow the electronics guide.' :
    item?.guide_template === 'tableware' ? 'Intact and still usable? Ordinary dirty tableware can be reused after washing. Tell us what the dirt is and whether it washes off.' :
    item?.guide_template === 'reuse' ? 'For reuse advice, confirm it\'s clean, intact, usable, and something you want to keep using.' :
    'An item\'s shape doesn\'t tell us its material. Confirm the material if you know it.';
}
function providerNote() {
  $('provider-note').textContent = $('provider').value === 'rules' ? 'The method comes straight from the guide rules and the condition you confirmed. Anything unknown is shown as a condition to check.' :
    $('provider').value === 'fixture' ? 'Simulation mode. Nothing is sent to Gemini, and every result is labeled as a simulation.' :
    config?.gemini_configured ? `We'll ask Gemini (${config.gemini_model}) directly. Your photo isn't sent.` :
      'Gemini isn\'t configured. Set GEMINI_API_KEY and GEMINI_MODEL in .env, then restart the demo.';
}
function renderRecommendation(result) {
  $('recommendation-empty').hidden = true; $('recommendation-result').hidden = false;
  const methodNames = {REDUCE:'Reduce',REUSE:'Reuse',RECYCLE:'Recycle',TRASH:'Trash'};
  $('method').textContent = result.status === 'READY' ? methodNames[result.recommended_method] : result.status === 'NEEDS_CONTEXT' ? 'Add details · starter advice' : result.special_handling ? 'Specialist handling advice' : 'Starter handling advice';
  const sourceNames = {rules:'Guide rules',fixture:'Simulation',gemini:'Gemini'};
  $('recommendation-source').textContent = result.status === 'READY' ? sourceNames[result.provider] || result.provider : 'General guidance with sources';
  $('reason').textContent = result.reason;
  $('recommendation-basis').hidden = result.status !== 'READY';
  $('recommendation-basis').textContent = result.recommendation_basis === 'CONDITIONAL' ?
    'Provisional advice · check the conditions before acting' : 'Based on the condition you described';
  $('reason-code').hidden = !result.reason_code;
  $('reason-code').textContent = messages[result.reason_code] || result.reason_code || '';
  if (result.provider === 'rules' && result.reason_code) $('reason-code').textContent = 'Advice is still available from the guide rules. '+$('reason-code').textContent;
  if (result.missing_context?.length) {
    const names = {'risk_type':'risk type','condition.cleanliness':'cleanliness','condition.integrity':'condition',
      'condition.reusable':'still usable','condition.cleanable':'washes off','condition.soiling':'dirt type',
      'condition.former_contents':'what it held',
      'reuse_intention':'want to reuse','recycling_access':'recycling access','object_key':'item type or material'};
    $('reason-code').textContent += ' Missing: '+result.missing_context.map(field => names[field] || field).join(', ')+'.';
  }
  $('recommendation-scope').hidden = !result.recommendation_scope;
  $('recommendation-scope').textContent = result.recommendation_scope === 'PREVENTION' ?
    'Prevention for next time. This isn\'t advice on getting rid of the item you have now.' :
    result.recommendation_basis === 'CONDITIONAL' ? 'A conditional handling plan. Anything left blank is still unknown.' : 'Handling for the item you confirmed.';
  $('handling-steps').replaceChildren();
  for (const step of result.steps) { const li = document.createElement('li'); li.textContent = step; $('handling-steps').append(li); }
  $('caution').hidden = !result.caution;
  $('caution').textContent = result.special_handling ? `Needs special handling: ${result.caution}` : result.caution;
  $('source-list').replaceChildren();
  for (const url of result.source_urls) {
    if (!url.startsWith('https://')) continue;
    const li = document.createElement('li'), link = document.createElement('a');
    const reference = result.source_details?.find(source => source.url === url);
    link.href = url; link.textContent = reference?.title || new URL(url).hostname + new URL(url).pathname; link.target = '_blank'; link.rel = 'noopener noreferrer';
    li.append(link);
    if (reference) {
      const note = document.createElement('p'); note.className = 'field-help';
      note.textContent = `${reference.publisher} · ${reference.region}${reference.checked_at ? ' · checked '+reference.checked_at : ''}. ${reference.scope_note}`;
      li.append(note);
    }
    $('source-list').append(li);
  }
  $('evidence-note').textContent = `${result.guide_status === 'DRAFT' ? 'Draft guide for the demo, not reviewed locally. ' : ''}${result.evidence_ids.length ? 'Evidence: '+result.evidence_ids.join(', ') : 'Starter advice from the general guide. Confirm the final method against the item\'s condition and your local collector.'}`;
  setStep(4);
}
async function requestRecommendation(event) {
  event.preventDefault();
  if (recommending || !$('object-key').value || !$('confirmed').checked || (!selected && !manualMode)) return;
  const version = recVersion;
  const intention = $('reuse-intention').value;
  const item = config.catalog.items.find(x => x.object_key === $('object-key').value);
  recRequestId ||= crypto.randomUUID(); recommending = true; updateForm();
  $('recommend').textContent = $('provider').value === 'gemini' ? 'Asking Gemini…' : $('provider').value === 'rules' ? 'Choosing a method…' : 'Running the simulation…';
  recController = new AbortController();
  const input = {client_request_id:recRequestId,detection_id:manualMode ? null : detection.detection_id,
    object_id:manualMode ? null : selected.object_id,manual:manualMode,confirmed_object:true,provider:$('provider').value,
    context:{object_key:item.object_key,category:item.category,condition:{cleanliness:$('cleanliness').value,
      integrity:$('integrity').value,reusable:$('reusable').value,cleanable:$('cleanable').value,soiling:$('soiling').value,former_contents:$('former-contents').value},reuse_intention:intention === '' ? null : intention === 'true',risk_flag:$('risk').checked,
      risk_type:$('risk-type').value,
      recommendation_goal:$('recommendation-goal').value,recycling_access:$('recycling-access').value}};
  try {
    const response = await api('recommend',{method:'POST',signal:recController.signal,headers:{'Content-Type':'application/json'},body:JSON.stringify(input)});
    if (version !== recVersion) return;
    renderRecommendation(response.data);
    if (response.data.status !== 'READY') recRequestId = null;
    message(response.data.status === 'READY' ? 'Your advice is ready. Check the sources and make sure they fit what you plan to do.' :
      'Starter advice is below. '+(messages[response.data.reason_code] || 'Check the condition and confirm a handler before you settle on a method.'),response.data.status === 'READY' ? '' : 'warning');
  } catch (error) {
    if (version === recVersion && error.name !== 'AbortError') {
      if (error.recommendation) renderRecommendation(error.recommendation);
      recRequestId = null;
      message(error.message,'error');
    }
  }
  finally { if (version === recVersion) { recommending = false; $('recommend').innerHTML = 'Get advice <span aria-hidden="true">↗</span>'; updateForm(); } }
}
function renderCatalog() {
  const query = $('catalog-search').value.trim().toLocaleLowerCase('en'); $('catalog').replaceChildren();
  for (const item of config.catalog.items) {
    const modelLabel = config.supported_object_labels[item.object_key];
    const supported = Boolean(modelLabel);
    if (catalogFilter === 'detected' && !supported || catalogFilter === 'manual' && supported || catalogFilter === 'special' && !item.special_handling) continue;
    if (query && !`${item.label} ${item.object_key} ${item.baseline_raw_class || ''} ${item.category}`.toLocaleLowerCase('en').includes(query)) continue;
    const button = document.createElement('button'); button.type = 'button'; button.className = 'catalog-item';
    const title = document.createElement('strong'), note = document.createElement('small'); title.textContent = item.label;
    note.textContent = item.special_handling ? 'Special handling' : !supported ? 'Pick manually · not in the active model' : `${rawLabel(modelLabel)} · confirm material`;
    button.append(title,note); button.addEventListener('click',() => chooseManual(item.object_key)); $('catalog').append(button);
  }
  if (!$('catalog').children.length) { const p = document.createElement('p'); p.textContent = 'No objects match your search.'; $('catalog').append(p); }
}
async function initialize() {
  try {
    config = await api('config');
    const supported = config.supported_visual_classes;
    $('engine-status').textContent = 'YOLO ready · '+(config.gemini_configured ? 'Gemini configured' : 'Gemini not configured');
    $('catalog-summary').textContent = `${supported} active visual classes cover ${Object.keys(config.supported_object_labels).length} of ${config.catalog.items.length} items and materials. You can pick the rest manually.`;
    $('model-coverage').textContent = `Active model: ${config.model_version}. ` +
      (config.combined_models ? 'Waste v3 plus general objects (including phones). Every guess still needs your confirmation, so pick the right type if it\'s wrong.' :
       config.supported_object_labels.cardboard ? 'Waste-only model: cardboard is covered, but phones and other general objects may not be. Pick manually or use the v3 launcher with the companion model.' :
        'This model has no cardboard class. For cardboard, pick the type manually or run demo-camera-v3.bat.');
    $('draft-label').textContent = config.demo_guides ? 'Draft guides · demo' : 'Reviewed guides only';
    fillObjectOptions([]); renderCatalog(); providerNote(); updateForm();
    // Running the demo opens the camera permission flow; upload still needs a click.
    if (new URLSearchParams(location.search).get('camera') !== 'off') await startCamera();
  } catch (error) { message(error.message,'error'); $('engine-status').textContent = 'Not connected'; }
}
$('start-camera').addEventListener('click',startCamera);
$('capture').addEventListener('click',capture);
$('detect').addEventListener('click',detectPhoto);
$('retake').addEventListener('click',async () => { await clearFlow(); await startCamera(); });
$('clear').addEventListener('click',clearFlow);
$('choose-file').addEventListener('click',() => $('file').click());
$('file').addEventListener('change',() => filePhoto($('file').files[0]));
$('manual').addEventListener('click',() => chooseManual());
$('context-form').addEventListener('submit',requestRecommendation);
$('object-key').addEventListener('change',() => { resetConditions(); clearRecommendation(); updateMaterialNote(); setStep(3); });
$('correct-detection').addEventListener('click',() => chooseManual($('object-key').value));
['cleanliness','integrity','reusable','cleanable','soiling','former-contents','reuse-intention','recommendation-goal','recycling-access','risk','risk-type','confirmed','provider'].forEach(id => $(id).addEventListener('change',() => {
  if (id === 'soiling' && $('soiling').value === 'HAZARDOUS') { $('risk').checked = true; $('risk-type').value = 'CHEMICAL'; }
  if (id === 'former-contents' && $('former-contents').value === 'CHEMICAL') { $('risk').checked = true; $('risk-type').value = 'CHEMICAL'; }
  updateConditionDetails(); clearRecommendation(); providerNote(); setStep(3);
}));
$('catalog-search').addEventListener('input',renderCatalog);
$('catalog-filters').addEventListener('click',event => { const button = event.target.closest('button[data-filter]'); if (!button) return; catalogFilter = button.dataset.filter; for (const b of $('catalog-filters').children) b.classList.toggle('active',b === button); renderCatalog(); });
document.addEventListener('visibilitychange',() => { if (document.hidden) stopCamera(); });
window.addEventListener('pagehide',() => { stopCamera(); detectController?.abort(); recController?.abort(); if (photoURL) URL.revokeObjectURL(photoURL); fetch(`${apiPrefix}session`,{method:'DELETE',headers:{'X-WasteNot-Demo':nonce},keepalive:true}).catch(() => {}); });
initialize();
