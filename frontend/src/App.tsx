import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

type Unit = { x: number | null; y: number | null; hp: number | null; max_hp: number; level?: number; gold?: number; alive?: boolean; visible?: boolean; lane?: string; respawn_timer?: number }
type MatchState = {
  time: number; game_over: boolean; winner: string | null
  map: { width: number; height: number; lane_y: number; top_lane_y?: number | null; bottom_lane_y?: number | null }
  champions: Record<'blue'|'red',Unit>; towers: Record<'blue'|'red',Unit>
  lane_towers: Record<'blue'|'red',Record<string,Unit & {alive:boolean}>>; bases: Record<'blue'|'red',Unit>
  minion_entities: Array<{team:string;x:number;y:number;hp:number;max_hp:number;lane?:string}>
  wards?: Array<{team:string;x:number;y:number;duration:number;max_duration:number}>
  wave_orders?: Record<string,Record<string,{mode:string;timer:number}>>
  objective: (Unit & {alive:boolean})|null; objective_respawn_timer: number|null
}
type ContextItem = { action:number; action_name:string; probability:number }
type Decision = { team:'blue'|'red'; action:number|null; action_name:string|null; target:{label:string;x?:number;y?:number}|null; observed_context:string[] }
type Match = { session_id:string; observation:number[]; action_mask:boolean[]; action_names:Record<string,string>; info:Record<string,unknown>; state:MatchState; steps:number; done:boolean; paused:boolean; decision_view:boolean; realtime:boolean; advanced_decisions:boolean; order_seconds:number; events:string[]; decisions:Record<'blue'|'red',Decision>; policy_preferences:ContextItem[] }
type HistoryItem = { step:number; team:'blue'|'red'; action:number|null; action_name:string; target:string }
type Experiment = { name:string; timesteps:number; overall_win_rate:number; opponents:Array<{opponent:string;win_rate:number;episodes:number;wins:number}> }

const API = import.meta.env.VITE_API_URL ?? 'http://localhost:8005'
const WS = API.replace(/^http/, 'ws')
const MACRO_ACTIONS = [17,18,19,20,21,22,23]
const ADVANCED_ACTIONS = [24,25,26,27,28,29,30,31,32,33,34,35]
const MACRO_TEXT:Record<number,{title:string;desc:string}> = {
  17:{title:'Engage enemy',desc:'Approach the rival champion and fight when in range.'},
  18:{title:'Push current lane',desc:'Advance on the tower in the lane you are occupying.'},
  19:{title:'Contest objective',desc:'Move to the neutral objective and attack it when possible.'},
  20:{title:'Farm lane',desc:'Secure nearby lane minions, then keep the lane moving.'},
  21:{title:'Recall and shop',desc:'Return to base to recover and choose an item upgrade.'},
  22:{title:'Defend current lane',desc:'Protect your tower and clear threats in your lane.'},
  23:{title:'Retreat to base',desc:'Disengage and return to safety.'},
  24:{title:'Rotate top',desc:'Leave your current lane to create pressure or help elsewhere.'},
  25:{title:'Rotate mid',desc:'Move through the center lane and contest central space.'},
  26:{title:'Rotate bottom',desc:'Shift to the bottom lane and change lane pressure.'},
  27:{title:'Freeze the wave',desc:'Hold allied minions near their current positions to farm safely.'},
  28:{title:'Slow push',desc:'Build a smaller advancing wave with controlled lane pressure.'},
  29:{title:'Fast push',desc:'Commit to a quick siege; allied minions move and hit harder.'},
  30:{title:'Ward objective',desc:'Spend gold and a ward charge to reveal the objective area.'},
  31:{title:'Sweep vision',desc:'Spend gold and a cooldown to remove nearby enemy wards.'},
  32:{title:'Zone objective',desc:'Hold space around the objective and punish visible challengers.'},
  33:{title:'All-in',desc:'Commit abilities and attacks to try to finish a visible rival.'},
  34:{title:'Poke',desc:'Use a ranged ability and keep distance instead of committing.'},
  35:{title:'Save cooldowns',desc:'Avoid nearby fights and take a safer farm opportunity.'},
}
const PRETTY:Record<string,string> = {
  wait:'Hold order',move_left:'Move toward left',move_right:'Move toward right',move_up:'Move toward top lane',move_down:'Move toward bottom lane',
  attack_champion:'Attack champion',attack_tower:'Attack tower',ability_1:'Use ability 1',ability_2:'Use ability 2',ability_3:'Use ability 3',
  attack_objective:'Attack objective',attack_base:'Attack base',farm_minion:'Farm minion',buy_blade:'Buy blade',buy_armor:'Buy armor',buy_boots:'Buy boots',buy_vitality:'Buy vitality',
}
const pretty = (name:string) => PRETTY[name] ?? name.replaceAll('_',' ')

function TacticalMap({state,decisions,step}:{state:MatchState;decisions:Match['decisions'];step:number}) {
  const W=1000,H=600,left=48,top=42,fieldW=904,fieldH=520
  const point=(x:number,y:number)=>({x:left+x/state.map.width*fieldW,y:top+(1-y/state.map.height)*fieldH})
  const lanes={top:state.map.top_lane_y??state.map.height*.25,mid:state.map.lane_y,bottom:state.map.bottom_lane_y??state.map.height*.75}
  const color=(team:string)=>team==='blue'?'#64b7ff':'#ff7684'
  const drawStructure=(unit:Unit,team:'blue'|'red',kind:'base'|'tower',lane='mid')=>{
    if(unit.x===null||unit.y===null)return null
    const p=point(unit.x,unit.y),size=kind==='base'?26:17,c=color(team),destroyed=unit.alive===false
    return <g key={`${team}-${kind}-${lane}`} opacity={destroyed?.32:1}>
      <rect x={p.x-size/2} y={p.y-size/2} width={size} height={size} rx={kind==='base'?6:3} fill={c} stroke="#f3f7fa" strokeWidth="1.4"/>
      {kind==='tower'&&<><rect x={p.x-20} y={p.y-size/2-7} width="40" height="4" rx="2" fill="#101821"/><rect x={p.x-20} y={p.y-size/2-7} width={40*Math.max(0,(unit.hp??0)/unit.max_hp)} height="4" rx="2" fill={c}/></>}
      <text x={p.x} y={p.y+size/2+12} textAnchor="middle" fill="#d8e1e9" fontSize="9">{kind==='base'?'CORE':lane.toUpperCase()}</text>
    </g>
  }
  const drawChampion=(unit:Unit,team:'blue'|'red')=>{
    if(unit.x===null||unit.y===null||unit.visible===false||unit.alive===false)return null
    const p=point(unit.x,unit.y),c=color(team),label=team==='blue'?'BLUE':'RED'
    return <g key={`${team}-champ`}>
      <circle cx={p.x} cy={p.y} r="13" fill={c} stroke="#f5f8fb" strokeWidth="2.5"/>
      <circle cx={p.x} cy={p.y} r="18" fill="none" stroke={c} strokeOpacity=".35" strokeWidth="1"/>
      <rect x={p.x-23} y={p.y-25} width="46" height="5" rx="2.5" fill="#111923"/>
      <rect x={p.x-23} y={p.y-25} width={46*Math.max(0,(unit.hp??0)/unit.max_hp)} height="5" rx="2.5" fill={c}/>
      <text x={p.x} y={p.y+31} textAnchor="middle" fill="#edf3f8" fontSize="10" fontWeight="700">{label} · L{unit.level??1}</text>
    </g>
  }
  const drawMoveCallout=(team:'blue'|'red')=>{
    const decision=decisions[team]
    if(!decision?.action_name)return null
    const unit=state.champions[team]
    const visible=unit.x!==null&&unit.y!==null&&unit.visible!==false
    const p=visible?point(unit.x!,unit.y!):{x:team==='blue'?145:855,y:team==='blue'?120:480}
    const width=176,height=48
    const x=Math.max(12,Math.min(W-width-12,team==='blue'?p.x+24:p.x-width-24))
    const calloutY=p.y<100?p.y+22:p.y-60
    const y=Math.max(36,Math.min(H-height-10,calloutY))
    const title=decision.action!==null?(MACRO_TEXT[decision.action]?.title??pretty(decision.action_name)):pretty(decision.action_name)
    const tone=team==='blue'?'#79c5ff':'#ff8190'
    return <g key={`${team}-${step}-${decision.action}`} className={`move-callout ${team}`} transform={`translate(${x} ${y})`} aria-label={`${team} used ${title}`}>
      <path className="move-callout-shadow" d="M7 3H161L176 18V43H15L0 28V10Z"/>
      <path className="move-callout-frame" d="M7 1H159L174 16V41H15L0 26V8Z"/>
      <path d="M7 1H56L42 5H9L4 10V7Z" fill={tone}/>
      <path d="M160 2L172 14V18L156 2Z" fill={tone} opacity=".75"/>
      <text x="13" y="17" className="move-callout-kicker" fill={tone}>{team==='blue'?'BLUE':'RED'} · CURRENT ORDER</text>
      <text x="13" y="34" className="move-callout-title">{title.toUpperCase()}</text>
      <path d="M157 28L166 28L160 36L151 36Z" fill={tone} opacity=".9"/>
    </g>
  }
  const drawOrder=(team:'blue'|'red')=>{
    const order=decisions[team],unit=state.champions[team],target=order?.target
    if(!target||target.x===undefined||target.y===undefined||unit.x===null||unit.y===null)return null
    const a=point(unit.x,unit.y),b=point(target.x,target.y),c=color(team)
    return <g key={`${team}-order`} pointerEvents="none"><line x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke={c} strokeWidth="2.4" strokeDasharray="8 7" opacity=".85" markerEnd={`url(#${team}-arrow)`}/><circle cx={b.x} cy={b.y} r="10" fill="#111923" fillOpacity=".8" stroke={c} strokeWidth="2"/></g>
  }
  return <div className="map-frame"><svg className="moba-map" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Three lane MOBA strategy map">
    <defs><linearGradient id="map-ground" x1="0" x2="1"><stop stopColor="#142a28"/><stop offset=".5" stopColor="#1b3730"/><stop offset="1" stopColor="#142a28"/></linearGradient><marker id="blue-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L10 5L0 10z" fill="#64b7ff"/></marker><marker id="red-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L10 5L0 10z" fill="#ff7684"/></marker></defs>
    <rect width={W} height={H} rx="16" fill="url(#map-ground)"/>
    {Object.entries(lanes).map(([lane,y])=>{const sy=point(0,y).y;const mode=state.wave_orders?.blue?.[lane]?.mode;return <g key={lane}><path d={`M${left} ${sy}H${left+fieldW}`} stroke="#665b45" strokeWidth="36"/><path d={`M${left} ${sy}H${left+fieldW}`} stroke="#c5b07f" strokeWidth="1.5" strokeDasharray="10 9" opacity=".68"/><text x="20" y={sy+4} fill="#decfae" fontSize="10" letterSpacing="1.4">{lane.toUpperCase()}</text>{mode&&mode!=='none'&&<text x="74" y={sy-11} fill="#f0d18c" fontSize="9">{mode.replace('_',' ').toUpperCase()}</text>}</g>})}
    <text x="55" y="25" fill="#8ebfe0" fontSize="10" letterSpacing="2">BLUE SIDE</text><text x="945" y="25" textAnchor="end" fill="#e7a0a6" fontSize="10" letterSpacing="2">RED SIDE</text>
    {Object.entries(state.lane_towers.blue??{}).map(([lane,t])=>drawStructure(t,'blue','tower',lane))}{Object.entries(state.lane_towers.red??{}).map(([lane,t])=>drawStructure(t,'red','tower',lane))}
    {drawStructure(state.bases.blue,'blue','base')}{drawStructure(state.bases.red,'red','base')}
    {state.objective?.alive&&state.objective.x!==null&&state.objective.y!==null&&(()=>{const p=point(state.objective.x!,state.objective.y!);return <g><circle cx={p.x} cy={p.y} r="14" fill="#d9ae5b" stroke="#ffedbd" strokeWidth="2"/><text x={p.x} y={p.y+4} textAnchor="middle" fill="#342812" fontSize="10" fontWeight="700">OBJ</text><rect x={p.x-24} y={p.y-23} width="48" height="4" fill="#211b10"/><rect x={p.x-24} y={p.y-23} width={48*state.objective.hp!/state.objective.max_hp} height="4" fill="#e7bf70"/></g>})()}
    {state.minion_entities.map((m,i)=>{const p=point(m.x,m.y);return <circle key={`${m.team}-${i}`} cx={p.x} cy={p.y} r="4.3" fill={color(m.team)} stroke="#101820" strokeWidth="1" opacity=".9"/>})}
    {(state.wards??[]).map((ward,i)=>{const p=point(ward.x,ward.y),c=color(ward.team);return <g key={`ward-${i}`}><circle cx={p.x} cy={p.y} r="8" fill="#101820" stroke={c} strokeWidth="2"/><circle cx={p.x} cy={p.y} r="17" fill="none" stroke={c} strokeOpacity=".5" strokeDasharray="3 3"/><title>{ward.team} ward · {Math.ceil(ward.duration)}s</title></g>})}
    {drawOrder('blue')}{drawOrder('red')}{drawChampion(state.champions.blue,'blue')}{drawChampion(state.champions.red,'red')}{drawMoveCallout('blue')}{drawMoveCallout('red')}
    {state.objective_respawn_timer!==null&&<text x="500" y="27" textAnchor="middle" fill="#e7c67d" fontSize="11">Objective returns in {state.objective_respawn_timer.toFixed(0)}s</text>}
    {state.game_over&&<g><rect x="340" y="257" width="320" height="76" rx="12" fill="#101820" fillOpacity=".94"/><text x="500" y="303" textAnchor="middle" fill="#f0d18c" fontSize="28" fontWeight="700">{state.winner?.toUpperCase()??'DRAW'} WINS</text></g>}
  </svg></div>
}

export default function App() {
  const [opponent,setOpponent]=useState('balanced')
  const [model,setModel]=useState('')
  const [redModel,setRedModel]=useState('')
  const [models,setModels]=useState<string[]>([])
  const [advancedModels,setAdvancedModels]=useState<string[]>([])
  const [experiments,setExperiments]=useState<Experiment[]>([])
  const [seed,setSeed]=useState(42)
  const [procedural,setProcedural]=useState(false)
  const [fog,setFog]=useState(false)
  const [advancedDecisions,setAdvancedDecisions]=useState(true)
  const [decisionView,setDecisionView]=useState(true)
  const [realtime,setRealtime]=useState(false)
  const [match,setMatch]=useState<Match|null>(null)
  const [history,setHistory]=useState<HistoryItem[]>([])
  const [sessionId,setSessionId]=useState('')
  const [status,setStatus]=useState('Connecting to local API')
  const [busy,setBusy]=useState(false)
  const [auto,setAuto]=useState(false)
  const [error,setError]=useState('')
  const socket=useRef<WebSocket|null>(null)
  useEffect(()=>{
    let alive=true,connected=false
    const checkApi=async()=>{
      try{
        const health=await fetch(`${API}/health`)
        if(!health.ok)throw new Error('API unavailable')
        if(!connected){
          const [modelResponse,experimentResponse]=await Promise.all([fetch(`${API}/models`),fetch(`${API}/experiments`)])
          const [modelData,experimentData]=await Promise.all([modelResponse.json(),experimentResponse.json()])
          if(alive){setModels(modelData.models??[]);setAdvancedModels(modelData.advanced_models??[]);setExperiments(experimentData.experiments??[])}
          connected=true
          if(alive)setStatus(previous=>previous==='Connecting to local API'||previous==='API offline'||previous==='API error'?'API ready':previous)
        }
      }catch{
        connected=false
        if(alive)setStatus('API offline')
      }
    }
    checkApi()
    const timer=window.setInterval(checkApi,5000)
    return()=>{alive=false;window.clearInterval(timer);socket.current?.close()}
  },[])
  const start=useCallback(async()=>{
    const autoStart=!!model&&(realtime||!decisionView||!!redModel)
    setError('');setAuto(autoStart);setHistory([])
    try{
      const response=await fetch(`${API}/sessions`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({opponent,seed:Number(seed),model:model||null,red_model:redModel||null,procedural_maps:procedural,fog_of_war:fog,decision_view:autoStart?false:decisionView,advanced_decisions:model?advancedModels.includes(model):redModel?advancedModels.includes(redModel):advancedDecisions,realtime})})
      const data=await response.json();if(!response.ok)throw new Error(data.detail??'Could not start match')
      socket.current?.close();setSessionId(data.session_id);setMatch(data);setStatus('Opening strategic match')
      const ws=new WebSocket(`${WS}/ws/sessions/${data.session_id}`);socket.current=ws
      ws.onopen=()=>setStatus('Strategic match ready')
      ws.onmessage=e=>{const next=JSON.parse(e.data);if(next.error){setError(next.error);setBusy(false);return}if(next.decisions){setHistory(previous=>{const rows=[...previous];for(const team of ['blue','red'] as const){const d=next.decisions[team] as Decision;if(d?.action_name){const target=d.target?.label??'';const prior=rows.find(row=>row.team===team);if(!prior||prior.action!==d.action||prior.target!==target){rows.unshift({step:next.steps,team,action:d.action,action_name:d.action_name,target})}}}return rows.slice(0,14)})}setMatch(next);setBusy(false)}
      ws.onerror=()=>setStatus('Match connection error')
      ws.onclose=()=>{if(socket.current===ws)setStatus('Match stream closed')}
    }catch(e){setError(e instanceof Error?e.message:String(e))}
  },[opponent,seed,model,redModel,procedural,fog,decisionView,advancedDecisions,advancedModels,realtime])
  const sendAction=(action:number)=>{if(busy||socket.current?.readyState!==WebSocket.OPEN)return;setBusy(true);socket.current.send(JSON.stringify({action}))}
  const stepAgent=()=>{if(busy||socket.current?.readyState!==WebSocket.OPEN)return;setBusy(true);socket.current.send(JSON.stringify({step_agent:true}))}
  const runAgent=()=>{if(busy||socket.current?.readyState!==WebSocket.OPEN)return;setAuto(true);setBusy(true);socket.current.send(JSON.stringify({use_agent:true,deterministic:true}))}
  const pause=()=>{setAuto(false);socket.current?.readyState===WebSocket.OPEN&&socket.current.send(JSON.stringify({pause:true}));setStatus('Paused between orders')}
  const resumeRealtime=()=>{socket.current?.readyState===WebSocket.OPEN&&socket.current.send(JSON.stringify({resume:true}));setStatus('Real-time match resumed')}
  const close=async()=>{setAuto(false);socket.current?.close();socket.current=null;if(sessionId)await fetch(`${API}/sessions/${sessionId}`,{method:'DELETE'}).catch(()=>undefined);setMatch(null);setSessionId('');setBusy(false);setStatus('API ready')}
  const downloadReplay=async()=>{if(!sessionId)return;try{const response=await fetch(`${API}/sessions/${sessionId}/replay`);if(!response.ok)throw new Error('Replay is not available');const blob=await response.blob();const url=URL.createObjectURL(blob);const link=document.createElement('a');link.href=url;link.download=`moba-replay-${sessionId.slice(0,12)}.json`;link.click();URL.revokeObjectURL(url)}catch(e){setError(e instanceof Error?e.message:String(e))}}
  const refreshExperiments=()=>fetch(`${API}/experiments`).then(r=>r.json()).then(d=>setExperiments(d.experiments??[])).catch(()=>undefined)
  const available=useMemo(()=>[...MACRO_ACTIONS,...(match?.advanced_decisions?ADVANCED_ACTIONS:[])].filter(i=>match?.action_mask[i]),[match?.action_mask,match?.advanced_decisions])
  const blue=match?.state.champions.blue,red=match?.state.champions.red
  const blueLane=blue?.lane??'mid'
  const blueTower=match?.state.lane_towers.blue?.[blueLane]
  const redTower=match?.state.lane_towers.red?.[blueLane]
  const health=(unit:Unit|undefined)=>unit?.hp===null||!unit?0:Math.max(0,100*unit.hp/(unit.max_hp||1))
  return <main className="page-shell">
    <header className="topbar"><a className="brand" href="#"><span className="brand-mark">M</span><span>MOBA<span className="brand-light">·RL</span></span></a><div className="topbar-meta"><span className="live-dot"/><span>{status}</span><span className="version-chip">STRATEGY ARENA</span></div></header>
    <section className="hero-row"><div><div className="eyebrow">THREE LANES · STRATEGIC SELF-PLAY</div><h1>Match <span>arena</span></h1><p>Issue a plan, watch it unfold, and inspect the map conditions around each decision.</p></div><div className="hero-stat"><span className="stat-label">MATCH CLOCK</span><strong>{match?`${Math.floor(match.state.time/60).toString().padStart(2,'0')}:${Math.floor(match.state.time%60).toString().padStart(2,'0')}`:'READY'}</strong><small>{match?(match.realtime?'Live pace · orders refresh every 0.5s':`Order interval · ${match.order_seconds}s`):'Start a strategic match'}</small></div></section>
    <div className="layout-grid">
      <aside className="panel setup-panel"><div className="panel-heading"><div><div className="eyebrow">MATCH SETUP</div><h2>New match</h2></div><span className="step-count">01</span></div>
        <label className="field-label">SCRIPTED OPPONENT</label><select value={opponent} onChange={e=>setOpponent(e.target.value)} disabled={!!sessionId||!!redModel}>{['balanced','aggressive','defensive','objective','adaptive'].map(name=><option key={name} value={name}>{name[0].toUpperCase()+name.slice(1)}</option>)}</select>
        <label className="field-label">OR TRAINED RED POLICY <span className="optional">OPTIONAL</span></label><select value={redModel} onChange={e=>{setRedModel(e.target.value);if(e.target.value){setFog(true);setAdvancedDecisions(advancedModels.includes(e.target.value))}}} disabled={!!sessionId}><option value="">Use scripted opponent above</option>{models.map(name=><option key={name} value={name}>{name}{advancedModels.includes(name)?' · advanced':''}</option>)}</select>
        <label className="field-label">BLUE POLICY <span className="optional">OPTIONAL</span></label><select value={model} onChange={e=>{setModel(e.target.value);if(e.target.value){setFog(true);setAdvancedDecisions(advancedModels.includes(e.target.value))}}} disabled={!!sessionId}><option value="">Human control</option>{models.map(name=><option key={name} value={name}>{name}{advancedModels.includes(name)?' · advanced':''}</option>)}</select>
        <label className="field-label">WORLD SEED</label><input type="number" value={seed} onChange={e=>setSeed(Number(e.target.value))} disabled={!!sessionId}/>
        <div className="toggle-stack">
          <label className="toggle-row"><span><b>Procedural map</b><small>Vary the seeded three-lane layout</small></span><input type="checkbox" checked={procedural} onChange={e=>setProcedural(e.target.checked)} disabled={!!sessionId}/></label>
          <label className="toggle-row"><span><b>Fog of war</b><small>Hide enemy units outside vision</small></span><input type="checkbox" checked={fog} onChange={e=>setFog(e.target.checked)} disabled={!!sessionId||!!model||!!redModel}/></label>
          {!model&&!redModel&&<label className="toggle-row"><span><b>Advanced decisions</b><small>Lane rotations, wave control, wards, and fight plans</small></span><input type="checkbox" checked={advancedDecisions} onChange={e=>setAdvancedDecisions(e.target.checked)} disabled={!!sessionId}/></label>}
          <label className="toggle-row"><span><b>Real-time combat</b><small>Game clock runs live; agents update plans every 0.5 seconds</small></span><input type="checkbox" checked={realtime} onChange={e=>setRealtime(e.target.checked)} disabled={!!sessionId}/></label>
          {!realtime&&<label className="toggle-row"><span><b>Decision view</b><small>Pause after each five-second order</small></span><input type="checkbox" checked={decisionView} onChange={e=>setDecisionView(e.target.checked)} disabled={!!sessionId}/></label>}
        </div>
        {!sessionId?<button className="primary-button" onClick={start}>Create match <span>↗</span></button>:<button className="secondary-button" onClick={close}>Close match</button>}{error&&<div className="error-box">{error}</div>}
        <div className="divider"/><div className="panel-heading small-heading"><div><div className="eyebrow">POLICY CONTROL</div><h2>Blue side</h2></div></div>
        <div className="control-buttons">{realtime?<><button className="primary-button compact" onClick={resumeRealtime} disabled={!sessionId||!match?.paused||!!match?.done}>Resume live <span>→</span></button><button className="secondary-button compact stop" onClick={pause} disabled={!sessionId||!!match?.paused||!!match?.done}>Pause</button></>:<><button className="primary-button compact" onClick={stepAgent} disabled={!sessionId||!model||!!match?.done||busy}>Next order <span>→</span></button>{!auto?<button className="secondary-button compact" onClick={runAgent} disabled={!sessionId||!model||!!match?.done||busy}>Run continuously</button>:<button className="secondary-button compact stop" onClick={pause}>Pause</button>}</>}</div>
        <div className="side-note">{realtime?'Live simulation updates every half-second; each policy commits to its plan for five seconds, matching its training interval. The order log records changes in plan.':model&&redModel?'Both policies are loaded. This match starts and runs automatically; use Pause if you want to inspect their decisions.':'Each order runs for five simulated seconds. The game pauses between orders so you can inspect the consequences.'}</div>
      </aside>
      <section className="panel match-panel"><div className="match-toolbar"><div className="match-heading"><div className="eyebrow">TACTICAL MAP</div><h2>{sessionId?'Live three-lane battlefield':'Arena preview'}</h2></div><div className="toolbar-chips"><span className="mode-chip">{procedural?'PROCEDURAL':'FIXED'} MAP</span><span className="mode-chip">{fog?'FOG ACTIVE':'FULL VISION'}</span><span className={`connection-chip ${match?.paused?'paused':''}`}><i/>{match?.paused?'PAUSED · CHOICE':sessionId?'LIVE':'IDLE'}</span>{match&&<button className="replay-button" onClick={downloadReplay}>Download replay</button>}</div></div>
        {match&&blue&&red?<>
          <div className="score-row"><div className="team-card blue-card"><span>BLUE · {model?'POLICY':'HUMAN'}</span><strong>{blue.hp?.toFixed(0)??'—'} <small>/ {blue.max_hp.toFixed(0)} HP</small></strong><div className="health-track"><i style={{width:`${health(blue)}%`}}/></div><small>LV {blue.level} · {blue.gold} gold · {blueLane.toUpperCase()}</small></div><div className="versus">VS</div><div className="team-card red-card"><span>RED · {redModel?'TRAINED POLICY':opponent.toUpperCase()}</span><strong>{red.hp?.toFixed(0)??'—'} <small>/ {red.max_hp.toFixed(0)} HP</small></strong><div className="health-track"><i style={{width:`${health(red)}%`}}/></div><small>LV {red.level} · {red.gold} gold · {red.lane?.toUpperCase()}</small></div></div>
          <TacticalMap state={match.state} decisions={match.decisions} step={match.steps}/>
          <section className="order-log" aria-label="Recent orders by side">
            {(['blue','red'] as const).map(team=>{
              const teamHistory=history.filter(row=>row.team===team).slice(0,4)
              return <div className={`order-log-team ${team}`} key={team}>
                <div className="order-log-heading"><span className="history-dot"/><b>{team.toUpperCase()}</b><small>RECENT ORDERS</small></div>
                <div className="order-log-list">{teamHistory.length?teamHistory.map((row,index)=><div className="order-log-row" key={`${row.team}-${row.step}-${row.action}-${index}`}>
                  <span className="order-log-step">{String(row.step).padStart(2,'0')}</span>
                  <span className="order-log-action">{MACRO_TEXT[row.action??-1]?.title??pretty(row.action_name)}{row.target&&<small>→ {row.target}</small>}</span>
                </div>):<div className="order-log-empty">Orders will appear as the match progresses.</div>}</div>
              </div>
            })}
          </section>
          <div className="lane-pressure"><div className="tower-pressure blue"><span className="pressure-label">BLUE {blueLane.toUpperCase()} TOWER</span><div className="pressure-track"><i className="blue-pressure" style={{width:`${health(blueTower)}%`}}/></div><b>{blueTower?.hp?.toFixed(0)??'—'} HP</b></div><div className="tower-pressure red"><span className="pressure-label">RED {blueLane.toUpperCase()} TOWER</span><div className="pressure-track"><i className="red-pressure" style={{width:`${health(redTower)}%`}}/></div><b>{redTower?.hp?.toFixed(0)??'—'} HP</b></div></div>
          <section className="decision-panel"><div className="decision-panel-head"><div><div className="eyebrow">CURRENT ORDERS</div><h3>Plan and visible game conditions</h3></div><span>{match.paused?'Paused after this decision':`Orders resolve over ${match.order_seconds} seconds`}</span></div><div className="decision-cards">{(['blue','red'] as const).map(team=>{const d=match.decisions[team];return <article className={`decision-card ${team}`} key={team}><div className="decision-team"><i/>{team.toUpperCase()} · {team==='blue'?(model?'POLICY':'PLAYER'):redModel?'TRAINED POLICY':'OPPONENT'}</div><strong>{d.action_name?(d.action!==null?(MACRO_TEXT[d.action]?.title??pretty(d.action_name)):pretty(d.action_name)):'Choosing first order'}</strong><p>{d.target?.label?`Target: ${d.target.label}`:'No target selected yet'}</p><ul>{(d.observed_context??[]).map((item,index)=><li key={index}>{item}</li>)}</ul></article>})}</div><p className="decision-note">Visible context reports game state. It does not claim to reveal the policy’s private reasoning.</p></section>
          {match.policy_preferences.length>0&&<section className="policy-panel"><div className="eyebrow">BLUE POLICY PREFERENCES</div><h3>Action probabilities for this state</h3><div className="probability-list">{match.policy_preferences.map(item=><div className="probability-row" key={item.action}><span>{MACRO_TEXT[item.action]?.title??pretty(item.action_name)}</span><div className="probability-track"><i style={{width:`${Math.max(1,item.probability*100)}%`}}/></div><b>{(item.probability*100).toFixed(1)}%</b></div>)}</div></section>}
          <div className="action-area"><div className="action-title"><div><div className="eyebrow">STRATEGIC ORDERS</div><strong>Choose Blue’s next plan</strong></div><span>{busy?'Order resolving…':match.done?'Match complete':`${available.length} macro choices`}</span></div><div className="order-grid">{MACRO_ACTIONS.map(id=><button key={id} className="order-card" disabled={!match.action_mask[id]||busy||!!match.done||!!model} onClick={()=>sendAction(id)}><span className="order-number">{String(id-16).padStart(2,'0')}</span><strong>{MACRO_TEXT[id].title}</strong><small>{MACRO_TEXT[id].desc}</small></button>)}</div>{match.advanced_decisions&&<><div className="action-title advanced-title"><div><div className="eyebrow">ADVANCED STRATEGY</div><strong>Higher risk, higher impact orders</strong></div><span>12 choices</span></div><div className="order-grid advanced-order-grid">{ADVANCED_ACTIONS.map(id=><button key={id} className="order-card advanced-order-card" disabled={!match.action_mask[id]||busy||!!match.done||!!model} onClick={()=>sendAction(id)}><span className="order-number">{String(id-16).padStart(2,'0')}</span><strong>{MACRO_TEXT[id].title}</strong><small>{MACRO_TEXT[id].desc}</small></button>)}</div><div className="action-title advanced-title"><div><div className="eyebrow">ITEM CHOICES</div><strong>Choose what to buy at base</strong></div><span>Trade gold for a lasting upgrade</span></div><div className="order-grid item-order-grid">{[13,14,15,16].map(id=><button key={id} className="order-card item-order-card" disabled={!match.action_mask[id]||busy||!!match.done||!!model} onClick={()=>sendAction(id)}><span className="order-number">{String(id-12).padStart(2,'0')}</span><strong>{({13:'Blade',14:'Armor',15:'Boots',16:'Vitality'} as Record<number,string>)[id]}</strong><small>{({13:'200 gold · increase attack damage',14:'150 gold · reduce incoming damage',15:'150 gold · move faster across the map',16:'200 gold · increase maximum health'} as Record<number,string>)[id]}</small></button>)}</div></>}<details className="tactical-details"><summary>Show tactical controls</summary><div className="action-grid">{Object.entries(match.action_names).map(([raw,name])=>{const id=Number(raw);if(id>=17)return null;return <button key={id} className="action-button" disabled={!match.action_mask[id]||busy||!!match.done||!!model} onClick={()=>sendAction(id)}><span className="action-key">{String(id).padStart(2,'0')}</span><span>{pretty(name)}</span></button>})}</div></details></div>
        </>:<div className="empty-arena"><div className="preview-map"><div className="preview-lane top"/><div className="preview-lane mid"/><div className="preview-lane bottom"/><span className="preview-base blue-base"/><span className="preview-objective">✦</span><span className="preview-base red-base"/></div><h3>Choose a strategy and create a match</h3><p>Each order has time to play out before the agents decide again.</p></div>}
      </section>
    </div>
    <section className="bottom-grid"><div className="panel objective-panel"><div className="eyebrow">NEUTRAL OBJECTIVE</div><h3>{match?.state.objective?.alive?'Available to contest':`Respawns in ${match?.state.objective_respawn_timer?.toFixed(0)??'—'}s`}</h3><p>Controlling the center grants gold and a team reward. Rotating here can leave a lane exposed.</p><div className="objective-bar"><i style={{width:`${match?.state.objective?100*(match.state.objective.hp??0)/match.state.objective.max_hp:0}%`}}/></div></div><div className="panel events-panel"><div className="events-head"><div><div className="eyebrow">MATCH EVENTS</div><h3>What changed</h3></div><span className="event-count">{match?.events.length??0}</span></div><div className="event-list">{match?.events.slice(-10).reverse().map((event,index)=><div className="event-row" key={`${index}-${event}`}><span className="event-marker"/>{event}</div>)??<div className="no-events">Minion, tower, objective, purchase, and combat events appear here.</div>}</div></div><div className="panel experiments-panel"><div className="events-head"><div><div className="eyebrow">EXPERIMENT TRACKING</div><h3>Latest evaluations</h3></div><button className="refresh-button" onClick={refreshExperiments}>Refresh</button></div>{experiments.length?experiments.slice(0,5).map(run=><article className="experiment-run" key={run.name}><div className="experiment-run-head"><b>{run.name}</b><span>{(run.overall_win_rate*100).toFixed(0)}% · {run.timesteps.toLocaleString()} steps</span></div><div className="experiment-opponents">{run.opponents.map(row=><span key={row.opponent}>{row.opponent} {(row.win_rate*100).toFixed(0)}%</span>)}</div></article>):<div className="no-events">No completed evaluation logs yet. Results appear after an evaluation checkpoint.</div>}</div></section>
    <footer><span>MOBA-RL · STRATEGIC ARENA</span><span>5-second macro orders · Seeded simulation</span></footer>
  </main>
}
