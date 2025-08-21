// ======================
// Parámetros y activaciones
// ======================
const CAPAS = [64, 16, 16, 10]; // 8x8 → 16 → 16 → 10
const RADIO_NEURONA = 8; // (compat)

// Escala de gauge por capa: [Entrada, Oculta1, Oculta2, Salida]
let GAUGE_SCALES = [1.0, 2.0, 2.5, 3.0]; // sube a 1.3–1.8 si quieres más grandes

// Duración de la animación de transición (ms)
let TWEEN_MS = 500; // ajustable en caliente con setTweenMs(ms)

const relu = x => Math.max(0, x);
const drelu = x => (x > 0 ? 1 : 0);
const softmax = (arr) => {
  const m = Math.max(...arr);
  const exps = arr.map(v => Math.exp(v - m));
  const s = exps.reduce((a,b)=>a+b, 0) || 1;
  return exps.map(v => v / s);
};

// ======================
// Utilidades
// ======================
function matrizAleatoria(filas, cols){
  const limite = Math.sqrt(6 / (filas + cols)); // Xavier simple
  return Array.from({length: filas}, () =>
    Array.from({length: cols}, () => (Math.random()*2 - 1) * limite)
  );
}
function vectorCeros(n){ return new Array(n).fill(0); }
function oneHot(k, n=10){ const v = vectorCeros(n); v[k]=1; return v; }
function dot(row, x){ let s = 0; for(let i=0;i<row.length;i++) s += row[i]*x[i]; return s; }
function clamp01(v){ return v < 0 ? 0 : (v > 1 ? 1 : v); }

// Color por peso para grids (rojo− / gris / azul+)
function colorPorPeso(w, maxAbs){
  const m = maxAbs || 1e-8;
  const t = Math.max(-1, Math.min(1, w / m)); // [-1,1]
  if (t >= 0){
    const b = 128 + Math.floor(127 * t);
    return `rgb(80,120,${b})`;
  } else {
    const r = 128 + Math.floor(127 * (-t));
    return `rgb(${r},90,90)`;
  }
}
function maxAbsArray(arr){
  let m = 0;
  for (let i=0;i<arr.length;i++) m = Math.max(m, Math.abs(arr[i]));
  return m || 1e-8;
}

// ======================
// Parámetros (con bias)
// ======================
let W1 = matrizAleatoria(16, 64), b1 = vectorCeros(16);
let W2 = matrizAleatoria(16, 16), b2 = vectorCeros(16);
let W3 = matrizAleatoria(10, 16), b3 = vectorCeros(10);

function reiniciarPesos(){
  W1 = matrizAleatoria(16, 64); b1 = vectorCeros(16);
  W2 = matrizAleatoria(16, 16); b2 = vectorCeros(16);
  W3 = matrizAleatoria(10, 16); b3 = vectorCeros(10);
  // recalcular a partir del lienzo actual y lanzar tween a nuevos objetivos
  hayCambios = true;
}

// ======================
// Lienzos y UI
// ======================
const lienzo = document.getElementById("lienzo");
const lctx = lienzo.getContext("2d");
const mini = document.getElementById("mini");
const mctx = mini.getContext("2d");
const chkEnVivo = document.getElementById("chk-en-vivo");
const rngPincel = document.getElementById("rng-pincel");
const selEtiqueta = document.getElementById("sel-etiqueta");
const elResultado = document.getElementById("resultado");
const contProbs = document.getElementById("probabilidades");
const lblDataset = document.getElementById("lbl-dataset");

// Feature grids canvases
const canvasH1 = document.getElementById("grid-h1");
const ctxH1 = canvasH1.getContext("2d");
const canvasH2 = document.getElementById("grid-h2");
const ctxH2 = canvasH2.getContext("2d");

function limpiarLienzo(){
  lctx.fillStyle = "#ffffff";
  lctx.fillRect(0,0,lienzo.width,lienzo.height);
}
limpiarLienzo();

let dibujando = false;
let tamPincel = parseInt(rngPincel.value,10);
let hayCambios = true;

lienzo.addEventListener("mousedown", e => { dibujando = true; pintar(e); hayCambios = true; });
lienzo.addEventListener("mouseup",   ()=> { dibujando = false; });
lienzo.addEventListener("mouseleave",()=> { dibujando = false; });
lienzo.addEventListener("mousemove", e => { if (dibujando){ pintar(e); hayCambios = true; }});
rngPincel.addEventListener("input", ()=> { tamPincel = parseInt(rngPincel.value,10); });

function pintar(e){
  const rect = lienzo.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;
  lctx.fillStyle = "#000000";
  lctx.beginPath();
  lctx.arc(x, y, tamPincel/2, 0, Math.PI*2);
  lctx.fill();
}

// Suavizado simple (box blur 3×3)
function suavizar(){
  const img = lctx.getImageData(0,0,lienzo.width,lienzo.height);
  const data = img.data;
  const w = lienzo.width, h = lienzo.height;
  const copia = new Uint8ClampedArray(data);
  const idx = (x,y,c) => ((y*w + x)<<2) + c;

  for(let y=1;y<h-1;y++){
    for(let x=1;x<w-1;x++){
      let sR=0,sG=0,sB=0,sA=0;
      for(let j=-1;j<=1;j++){
        for(let i=-1;i<=1;i++){
          const k = idx(x+i, y+j, 0);
          sR += copia[k+0]; sG += copia[k+1]; sB += copia[k+2]; sA += copia[k+3];
        }
      }
      const o = idx(x,y,0);
      data[o+0] = (sR/9)|0;
      data[o+1] = (sG/9)|0;
      data[o+2] = (sB/9)|0;
      data[o+3] = (sA/9)|0;
    }
  }
  lctx.putImageData(img,0,0);
  hayCambios = true;
}

// ======================
// Entrada 8×8 (y vista ampliada)
// ======================
function entrada8x8(){
  const tmp = document.createElement("canvas");
  tmp.width = 8; tmp.height = 8;
  const tctx = tmp.getContext("2d");
  tctx.drawImage(lienzo, 0, 0, 8, 8);
  const datos = tctx.getImageData(0,0,8,8).data;

  const entrada = [];
  for (let i=0; i<datos.length; i+=4){
    const gris = (datos[i]+datos[i+1]+datos[i+2])/3;
    entrada.push(1 - gris/255); // negro→1, blanco→0
  }

  mctx.imageSmoothingEnabled = false;
  mctx.clearRect(0,0,mini.width,mini.height);
  mctx.drawImage(tmp, 0, 0, 128, 128);

  return entrada;
}

// ======================
// Forward + Backprop
// ======================
let ultimaEntrada   = new Array(64).fill(0);
let ultimasProbs    = new Array(10).fill(0);
let ultH1           = new Array(16).fill(0);
let ultH2           = new Array(16).fill(0);

function forward(x){
  // Capa 1
  const z1 = new Array(16);
  for(let i=0;i<16;i++) z1[i] = dot(W1[i], x) + b1[i];
  const a1 = z1.map(relu);
  // Capa 2
  const z2 = new Array(16);
  for(let i=0;i<16;i++) z2[i] = dot(W2[i], a1) + b2[i];
  const a2 = z2.map(relu);
  // Salida
  const z3 = new Array(10);
  for(let i=0;i<10;i++) z3[i] = dot(W3[i], a2) + b3[i];
  const probs = softmax(z3);
  return { z1, a1, z2, a2, z3, probs };
}

function inferir(entrada){
  const { z1, a1, z2, a2, z3, probs } = forward(entrada);
  ultimaEntrada = entrada.slice();
  ultimasProbs  = probs.slice();
  ultH1 = a1.slice();
  ultH2 = a2.slice();
  return { a1, a2, probs };
}

function backprop(x, yIndex, lr=0.1, l2=0.0){
  const y = oneHot(yIndex, 10);
  const { z1, a1, z2, a2, z3, probs } = forward(x);

  // dL/dz3 = probs - y
  const dz3 = probs.map((p,i)=> p - y[i]);

  // Gradientes W3, b3
  const dW3 = Array.from({length:10},()=>Array(16).fill(0));
  const db3_local = dz3.slice();
  for(let i=0;i<10;i++){
    for(let j=0;j<16;j++) dW3[i][j] = dz3[i] * a2[j];
  }

  // dL/da2
  const da2 = new Array(16).fill(0);
  for(let j=0;j<16;j++){
    for(let i=0;i<10;i++) da2[j] += dz3[i]*W3[i][j];
  }
  const dz2 = da2.map((v,i)=> v * drelu(z2[i]));

  // Gradientes W2, b2
  const dW2 = Array.from({length:16},()=>Array(16).fill(0));
  const db2_local = dz2.slice();
  for(let i=0;i<16;i++){
    for(let j=0;j<16;j++) dW2[i][j] = dz2[i] * a1[j];
  }

  // dL/da1
  const da1 = new Array(16).fill(0);
  for(let j=0;j<16;j++){
    for(let i=0;i<16;i++) da1[j] += dz2[i]*W2[i][j];
  }
  const dz1 = da1.map((v,i)=> v * drelu(z1[i]));

  // Gradientes W1, b1
  const dW1 = Array.from({length:16},()=>Array(64).fill(0));
  const db1_local = dz1.slice();
  for(let i=0;i<16;i++){
    for(let j=0;j<64;j++) dW1[i][j] = dz1[i] * x[j];
  }

  // Regularización L2
  if(l2>0){
    for(let i=0;i<16;i++) for(let j=0;j<64;j++) dW1[i][j] += l2*W1[i][j];
    for(let i=0;i<16;i++) for(let j=0;j<16;j++) dW2[i][j] += l2*W2[i][j];
    for(let i=0;i<10;i++) for(let j=0;j<16;j++) dW3[i][j] += l2*W3[i][j];
  }

  // Actualización SGD
  for(let i=0;i<16;i++){
    for(let j=0;j<64;j++) W1[i][j] -= lr * dW1[i][j];
    b1[i] -= lr * db1_local[i];
  }
  for(let i=0;i<16;i++){
    for(let j=0;j<16;j++) W2[i][j] -= lr * dW2[i][j];
    b2[i] -= lr * db2_local[i];
  }
  for(let i=0;i<10;i++){
    for(let j=0;j<16;j++) W3[i][j] -= lr * dW3[i][j];
    b3[i] -= lr * db3_local[i];
  }

  return -Math.log(probs[yIndex] + 1e-9); // pérdida CE
}

// ======================
// Probabilidades UI
// ======================
function pintarProbabilidades(probs){
  contProbs.innerHTML = "";
  probs.forEach((p, i)=>{
    const fila = document.createElement("div");
    fila.className = "fila-prob";

    const etiqueta = document.createElement("div");
    etiqueta.textContent = i.toString();

    const env = document.createElement("div");
    env.className = "barra-env";

    const barra = document.createElement("div");
    barra.className = "barra";
    barra.style.width = (p*100).toFixed(1) + "%";
    env.appendChild(barra);

    const porc = document.createElement("div");
    porc.textContent = (p*100).toFixed(1) + "%";

    fila.appendChild(etiqueta);
    fila.appendChild(env);
    fila.appendChild(porc);
    contProbs.appendChild(fila);
  });
}

// ======================
// Visualización de red (pesos y activaciones) — GAUGES
// ======================
const lienzoRed = document.getElementById("red");
const rctx = lienzoRed.getContext("2d");

// Radios base por capa y multiplicadores
function radioPorCapa(li){
  const base = [6, 10, 10, 12]; // entrada, oculta1, oculta2, salida
  const scale = GAUGE_SCALES[li] ?? 1.0;
  return Math.max(3, Math.round(base[li] * scale));
}

// Dibuja un gauge radial con aguja para un valor v ∈ [0,1]
function dibujarGauge(x, y, r, v){
  const start = -Math.PI * 5/6;  // -150°
  const end   =  Math.PI * 5/6;  // +150°
  const ang   = start + clamp01(v) * (end - start);

  rctx.save();
  rctx.translate(x, y);

  // Fondo del gauge
  rctx.beginPath();
  rctx.arc(0, 0, r, 0, Math.PI*2);
  rctx.fillStyle = "#0b0e1b";
  rctx.fill();

  // Aro exterior
  rctx.lineWidth = 1.5;
  rctx.strokeStyle = "#c9d2ff22";
  rctx.beginPath();
  rctx.arc(0, 0, r, 0, Math.PI*2);
  rctx.stroke();

  // Arco base (escala)
  rctx.lineWidth = Math.max(2, Math.round(r*0.25/3));
  rctx.strokeStyle = "rgba(180,190,255,0.28)";
  rctx.beginPath();
  rctx.arc(0, 0, r - 2, start, end);
  rctx.stroke();

  // Arco de valor
  rctx.strokeStyle = "rgba(98,186,255,0.85)";
  rctx.beginPath();
  rctx.arc(0, 0, r - 2, start, ang);
  rctx.stroke();

  // Aguja
  const nx = Math.cos(ang) * (r - 3);
  const ny = Math.sin(ang) * (r - 3);
  rctx.lineWidth = Math.max(1.5, r*0.18);
  rctx.strokeStyle = "#8ec5ff";
  rctx.beginPath();
  rctx.moveTo(0, 0);
  rctx.lineTo(nx, ny);
  rctx.stroke();

  // Núcleo
  rctx.beginPath();
  rctx.arc(0, 0, Math.max(1.2, r * 0.18), 0, Math.PI*2);
  rctx.fillStyle = "#0b0e1b";
  rctx.fill();
  rctx.lineWidth = 1.2;
  rctx.strokeStyle = "#c9d2ff44";
  rctx.stroke();

  rctx.restore();
}

// ======================
// Tween de activaciones (animación)
// ======================

// Easing (rápido y suave)
function easeOutCubic(t){ return 1 - Math.pow(1 - t, 3); }

// Estados "mostrados" en gauges (interpolados)
let dispEntrada = new Array(64).fill(0);
let dispH1      = new Array(16).fill(0);
let dispH2      = new Array(16).fill(0);
let dispSalida  = new Array(10).fill(0);

// Origenes y objetivos del tween
let startEntrada = dispEntrada.slice();
let startH1      = dispH1.slice();
let startH2      = dispH2.slice();
let startSalida  = dispSalida.slice();

let tgtEntrada = dispEntrada.slice();
let tgtH1      = dispH1.slice();
let tgtH2      = dispH2.slice();
let tgtSalida  = dispSalida.slice();

let tweenStartTime = 0;
let animating = false;

function setGaugeTargets(entrada, a1, a2, probs){
  // copiar estados actuales como origen
  startEntrada = dispEntrada.slice();
  startH1      = dispH1.slice();
  startH2      = dispH2.slice();
  startSalida  = dispSalida.slice();

  // establecer objetivos
  tgtEntrada = entrada.slice();
  tgtH1      = a1.slice();
  tgtH2      = a2.slice();
  tgtSalida  = probs.slice();

  tweenStartTime = performance.now();
  animating = true;
}

function stepTween(now){
  if (!animating) return false;
  const t = clamp01((now - tweenStartTime) / TWEEN_MS);
  const e = easeOutCubic(t);

  // interpolación
  for (let i=0;i<dispEntrada.length;i++){
    dispEntrada[i] = startEntrada[i] + (tgtEntrada[i]-startEntrada[i]) * e;
  }
  for (let i=0;i<dispH1.length;i++){
    dispH1[i] = startH1[i] + (tgtH1[i]-startH1[i]) * e;
  }
  for (let i=0;i<dispH2.length;i++){
    dispH2[i] = startH2[i] + (tgtH2[i]-startH2[i]) * e;
  }
  for (let i=0;i<dispSalida.length;i++){
    dispSalida[i] = startSalida[i] + (tgtSalida[i]-startSalida[i]) * e;
  }

  if (t >= 1){
    // asegurar aterrizaje exacto
    dispEntrada = tgtEntrada.slice();
    dispH1 = tgtH1.slice();
    dispH2 = tgtH2.slice();
    dispSalida = tgtSalida.slice();
    animating = false;
  }
  return true;
}

function dibujarRedDesdeDisp(){
  // conexiones dependen de pesos; gauges usan los valores interpolados
  const w = lienzoRed.width, h = lienzoRed.height;
  rctx.clearRect(0,0,w,h);

  const capas = CAPAS.slice();
  const xs = [];
  const pasoX = w / (capas.length + 1);
  for (let i=0; i<capas.length; i++) xs.push(pasoX*(i+1));

  const posiciones = [];
  for (let li=0; li<capas.length; li++){
    const n = capas[li];
    const pasoY = h / (n + 1);
    const capaPos = [];
    for (let j=0; j<n; j++) capaPos.push({ x: xs[li], y: pasoY*(j+1) });
    posiciones.push(capaPos);
  }

  const maxAbs = (W) => {
    let m = 0;
    for (let r=0; r<W.length; r++)
      for (let c=0; c<W[r].length; c++)
        m = Math.max(m, Math.abs(W[r][c]));
    return m || 1;
  };
  const m1 = maxAbs(W1), m2 = maxAbs(W2), m3 = maxAbs(W3);

  function dibujarConexiones(W, desde, hasta, maxW){
    const BASE = 0.3, ESCALA = 4.0;
    for (let r=0; r<W.length; r++){
      for (let c=0; c<W[r].length; c++){
        const w = W[r][c];
        const p1 = desde[c], p2 = hasta[r];
        const t = Math.abs(w)/maxW;
        const ancho = BASE + t*ESCALA;
        const alfa = 0.15 + 0.55*t;

        rctx.beginPath();
        rctx.moveTo(p1.x, p1.y);
        rctx.lineTo(p2.x, p2.y);
        rctx.lineWidth = ancho;
        rctx.strokeStyle = (w >= 0)
          ? `rgba(100,181,246,${alfa})`
          : `rgba(239,83,80,${alfa})`;
        rctx.stroke();
      }
    }
  }

  dibujarConexiones(W1, posiciones[0], posiciones[1], m1);
  dibujarConexiones(W2, posiciones[1], posiciones[2], m2);
  dibujarConexiones(W3, posiciones[2], posiciones[3], m3);

  function dibujarNeuronasComoGauge(posCapa, activaciones=[], li){
    const r = radioPorCapa(li);
    for (let i=0; i<posCapa.length; i++){
      const {x,y} = posCapa[i];
      const a = clamp01(activaciones[i] ?? 0);
      dibujarGauge(x, y, r, a);
    }
  }

  dibujarNeuronasComoGauge(posiciones[0], dispEntrada, 0);
  dibujarNeuronasComoGauge(posiciones[1], dispH1, 1);
  dibujarNeuronasComoGauge(posiciones[2], dispH2, 2);
  dibujarNeuronasComoGauge(posiciones[3], dispSalida, 3);
}

// ======================
// Feature grids (H1 y H2)
// ======================
function dibujarGridH1(a1){
  const tiles = 4;      // 4×4 = 16 neuronas
  const cellPix = 8;    // cada neurona = 8×8
  const gap = 2;
  const tileSize = Math.floor((canvasH1.width - (tiles-1)*gap) / tiles);
  ctxH1.clearRect(0,0,canvasH1.width, canvasH1.height);

  for (let n=0;n<16;n++){
    const row = Math.floor(n / tiles);
    const col = n % tiles;
    const x0 = col * (tileSize + gap);
    const y0 = row * (tileSize + gap);

    const wRow = W1[n];
    const mabs = maxAbsArray(wRow);
    const scale = tileSize / cellPix;

    for (let py=0; py<cellPix; py++){
      for (let px=0; px<cellPix; px++){
        const w = wRow[py*cellPix + px];
        ctxH1.fillStyle = colorPorPeso(w, mabs);
        ctxH1.fillRect(x0 + px*scale, y0 + py*scale, Math.ceil(scale), Math.ceil(scale));
      }
    }

    // overlay según activación (usamos dispH1 al dibujar en vivo)
    const act = Math.max(0, Math.min(1, a1[n] || 0));
    ctxH1.fillStyle = `rgba(255,255,255,${0.12 + 0.18*act})`;
    ctxH1.fillRect(x0, y0, tileSize, tileSize);

    ctxH1.lineWidth = 1;
    ctxH1.strokeStyle = `rgba(200,210,255,${0.3 + 0.7*act})`;
    ctxH1.strokeRect(x0 + 0.5, y0 + 0.5, tileSize - 1, tileSize - 1);
  }
}

function dibujarGridH2(a2){
  const tiles = 4;  // 4×4 neuronas
  const cell = 4;   // cada neurona mostrada como 4×4 (pesos hacia H1)
  const gap = 2;
  const tileSize = Math.floor((canvasH2.width - (tiles-1)*gap) / tiles);
  ctxH2.clearRect(0,0,canvasH2.width, canvasH2.height);

  for (let n=0;n<16;n++){
    const row = Math.floor(n / tiles);
    const col = n % tiles;
    const x0 = col * (tileSize + gap);
    const y0 = row * (tileSize + gap);

    const wRow = W2[n];             // 16 pesos hacia H1
    const mabs = maxAbsArray(wRow);
    const scale = tileSize / cell;

    for (let py=0; py<cell; py++){
      for (let px=0; px<cell; px++){
        const idx = py*cell + px;
        const w = wRow[idx];
        ctxH2.fillStyle = colorPorPeso(w, mabs);
        ctxH2.fillRect(x0 + px*scale, y0 + py*scale, Math.ceil(scale), Math.ceil(scale));
      }
    }

    const act = Math.max(0, Math.min(1, a2[n] || 0));
    ctxH2.fillStyle = `rgba(255,255,255,${0.12 + 0.18*act})`;
    ctxH2.fillRect(x0, y0, tileSize, tileSize);

    ctxH2.lineWidth = 1;
    ctxH2.strokeStyle = `rgba(200,210,255,${0.3 + 0.7*act})`;
    ctxH2.strokeRect(x0 + 0.5, y0 + 0.5, tileSize - 1, tileSize - 1);
  }
}

function dibujarGridsCaracteristicas(a1=[], a2=[]){
  dibujarGridH1(a1);
  dibujarGridH2(a2);
}

// ======================
// Predicción en vivo + bucle de animación
// ======================
function actualizarEnVivo(now){
  if (chkEnVivo.checked && hayCambios){
    const entrada = entrada8x8();
    const { a1, a2, probs } = inferir(entrada);
    const pred = probs.indexOf(Math.max(...probs));

    elResultado.textContent = pred.toString();
    pintarProbabilidades(probs);

    // establecer nuevos objetivos de tween (gauges y grids)
    setGaugeTargets(entrada, a1, a2, probs);

    // grids usan activaciones para overlay (que también interpolamos al dibujar)
    hayCambios = false;
  }

  // avanzar tween (si hay)
  const dirty = stepTween(performance.now());

  // dibujar red y grids desde valores mostrados (interpolados)
  dibujarRedDesdeDisp();
  dibujarGridsCaracteristicas(dispH1, dispH2);

  requestAnimationFrame(actualizarEnVivo);
}
requestAnimationFrame(actualizarEnVivo);

// ======================
// Dataset y entrenamiento
// ======================
const dataset = []; // {x:[64], y:0..9}

function agregarEjemploActual(){
  const x = entrada8x8();
  const y = parseInt(selEtiqueta.value, 10);
  dataset.push({x, y});
  lblDataset.textContent = dataset.length.toString();
}

function entrenarPasoActual(lr, l2){
  const x = entrada8x8();
  const y = parseInt(selEtiqueta.value, 10);
  backprop(x, y, lr, l2);
  hayCambios = true;
}

function entrenarDataset(pasos, lr, l2){
  if (dataset.length === 0) return;
  for(let k=0;k<pasos;k++){
    const {x, y} = dataset[Math.floor(Math.random()*dataset.length)];
    backprop(x, y, lr, l2);
  }
  hayCambios = true;
}

// ======================
// Botones
// ======================
document.getElementById("btn-limpiar").addEventListener("click", ()=>{
  limpiarLienzo();
  elResultado.textContent = "?";
  // setters de objetivos a cero para un tween hacia apagado
  setGaugeTargets(new Array(64).fill(0), new Array(16).fill(0), new Array(16).fill(0), new Array(10).fill(0));
  pintarProbabilidades(new Array(10).fill(0));
  hayCambios = false; // ya se encarga el tween
});
document.getElementById("btn-suavizar").addEventListener("click", ()=> suavizar());
document.getElementById("btn-reiniciar-pesos").addEventListener("click", ()=> reiniciarPesos());
document.getElementById("btn-agregar-ejemplo").addEventListener("click", ()=> agregarEjemploActual());

document.getElementById("btn-train-1").addEventListener("click", ()=>{
  const lr = parseFloat(document.getElementById("inp-lr").value);
  const l2 = parseFloat(document.getElementById("inp-l2").value);
  entrenarPasoActual(lr, l2);
});
document.getElementById("btn-train-50").addEventListener("click", ()=>{
  const lr = parseFloat(document.getElementById("inp-lr").value);
  const l2 = parseFloat(document.getElementById("inp-l2").value);
  entrenarDataset(50, lr, l2);
});
document.getElementById("btn-train-500").addEventListener("click", ()=>{
  const lr = parseFloat(document.getElementById("inp-lr").value);
  const l2 = parseFloat(document.getElementById("inp-l2").value);
  entrenarDataset(500, lr, l2);
});

// ======================
// Controles (consola/atajos)
// ======================
// Cambia una capa concreta: 0=Entrada, 1=Oculta1, 2=Oculta2, 3=Salida
window.setLayerGaugeScale = function(layer, scale){
  const l = Math.max(0, Math.min(3, Math.floor(layer)));
  const s = Math.max(0.5, Math.min(3, Number(scale) || 1.0));
  GAUGE_SCALES[l] = s;
};
window.setAllGaugeScales = function(s0=1, s1=1, s2=1, s3=1){
  GAUGE_SCALES = [
    Math.max(0.5, Math.min(3, Number(s0)||1)),
    Math.max(0.5, Math.min(3, Number(s1)||1)),
    Math.max(0.5, Math.min(3, Number(s2)||1)),
    Math.max(0.5, Math.min(3, Number(s3)||1)),
  ];
};
// Duración del tween en caliente
window.setTweenMs = function(ms){
  TWEEN_MS = Math.max(0, Math.min(5000, Number(ms)||500));
};

// Atajos de teclado: Ctrl+1..4 para aumentar; Ctrl+Alt+1..4 para reducir
document.addEventListener('keydown', (e)=>{
  const keyToLayer = { '1':0, '2':1, '3':2, '4':3 };
  if (keyToLayer.hasOwnProperty(e.key)){
    const layer = keyToLayer[e.key];
    if (e.ctrlKey && !e.altKey){
      GAUGE_SCALES[layer] = Math.min(3, (GAUGE_SCALES[layer] + 0.1));
    } else if (e.ctrlKey && e.altKey){
      GAUGE_SCALES[layer] = Math.max(0.5, (GAUGE_SCALES[layer] - 0.1));
    }
  }
});

// ======================
// Pintado inicial (estado reposo)
// ======================
pintarProbabilidades(new Array(10).fill(0));
// arranque con tween suave desde 0 a 0 (no visible) y primer render
setGaugeTargets(new Array(64).fill(0), new Array(16).fill(0), new Array(16).fill(0), new Array(10).fill(0));

