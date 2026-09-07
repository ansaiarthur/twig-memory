/**
 * 实时 LLM 判定层（Moonshot / Kimi API）
 * 请求经 vite dev 代理 /moonshot → api.moonshot.cn，密钥由代理附加，浏览器不可见。
 * 任何失败（超时 / 网络 / 解析）由调用方回退到预计算脚本 —— 设计债务⑩的现场稳定性策略。
 */

const TIMEOUT_MS = 10000
/** 主模型随 MUNINN_MODEL（与 server/llm-node.ts 同口径，缺省现役 kimi-k2.6）。
 *  浏览器侧无 process 全局，走 globalThis 探测，取不到即落缺省 */
const MODEL = (globalThis as { process?: { env?: Record<string, string> } }).process?.env?.MUNINN_MODEL || 'kimi-k2.6'

export interface ChatMessage { role: 'system' | 'user' | 'assistant'; content: string }

/** 可注入的聊天传输层：服务端（Node 直连）注入自定义实现；前端不注入时走 vite 代理，行为完全不变。
 *  opts.model 为按调用覆盖的模型名（异源反证生成用，缺省用默认模型） */
export type ChatTransport = (
  messages: ChatMessage[],
  opts?: { temperature?: number; maxTokens?: number; model?: string },
) => Promise<string>

let customTransport: ChatTransport | null = null

export function setChatTransport(t: ChatTransport | null): void {
  customTransport = t
}

export async function moonshotChat(
  messages: ChatMessage[],
  opts?: { temperature?: number; maxTokens?: number; model?: string },
): Promise<string> {
  if (customTransport) return customTransport(messages, opts)
  const ctrl = new AbortController()
  const timer = setTimeout(() => ctrl.abort(), TIMEOUT_MS)
  try {
    const resp = await fetch('/moonshot/v1/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: opts?.model ?? MODEL,
        temperature: opts?.temperature ?? 0.3,
        max_tokens: opts?.maxTokens ?? 700,
        messages,
      }),
      signal: ctrl.signal,
    })
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
    const data = await resp.json()
    const text = data?.choices?.[0]?.message?.content
    if (typeof text !== 'string' || !text.trim()) throw new Error('空响应')
    return text
  } finally {
    clearTimeout(timer)
  }
}

/**
 * 从模型输出中提取第一个 JSON 对象。
 * P2-7 修复：平衡括号扫描替代贪婪正则——贪婪正则 \{[\s\S]*\} 在模型输出含多个 {} 时
 * 会捕获从首个 { 到末个 } 的整段（含中间非 JSON 文本），导致 parse 失败。
 * 平衡扫描正确处理嵌套、字符串内的括号、转义字符。
 */
export function extractJson<T>(raw: string): T | null {
  const start = raw.indexOf('{')
  if (start < 0) return null
  let depth = 0, inStr = false, esc = false
  for (let i = start; i < raw.length; i++) {
    const c = raw[i]
    if (esc) { esc = false; continue }
    if (c === '\\') { esc = true; continue }
    if (c === '"') { inStr = !inStr; continue }
    if (inStr) continue
    if (c === '{') depth++
    else if (c === '}') {
      depth--
      if (depth === 0) {
        try { return JSON.parse(raw.slice(start, i + 1)) as T } catch { return null }
      }
    }
  }
  return null
}

const clamp01 = (v: unknown): number | null =>
  typeof v === 'number' && isFinite(v) ? Math.min(0.95, Math.max(0.05, v)) : null

/* ---------- 判定一：矛盾响应（人为反例） ---------- */

export interface CounterVerdict {
  conflictType: string   // 事实冲突 / 偏好反转 / 能力变化 / 关系变化 / 无冲突
  hasConflict: boolean
  conviction: number     // 新置信度
  revised: string        // 加限定语后的论断
  reply: string          // 对用户说的话：诚实承认不确定，温暖，不评判
}

export async function adjudicateCounter(claimText: string, conviction: number, userStatement: string): Promise<CounterVerdict | null> {
  const raw = await moonshotChat([
    {
      role: 'system',
      content: `你是「雾尼」叙事记忆引擎的矛盾响应判定模块。系统对用户的既有理解会以论断形式保存，并挂置信度。当用户说出与既有论断冲突的话时，你必须：1) 判断冲突类型学（事实冲突/偏好反转/能力变化/关系变化/无冲突）；2) 给出修正后的新置信度——冲突越直接，下调越多；3) 给论断加限定语重写，保留长期观察但承认新变化；4) 写一段以系统口吻对用户说的话：明确承认「我不那么确定了」，给出可能原因猜测，温暖、不评判、不说教、不超过 80 字。重要：用户陈述与论断一致或只是日常琐事细节时，必须判无冲突（hasConflict:false）——不要把中性陈述过度解读为冲突。只输出 JSON：{"conflictType":"...","hasConflict":true,"conviction":0.57,"revised":"...","reply":"..."}`,
    },
    {
      role: 'user',
      content: `既有论断：「${claimText}」（当前置信度 ${conviction.toFixed(2)}）\n用户新陈述：「${userStatement}」`,
    },
  ], { temperature: 0.4 })
  const j = extractJson<CounterVerdict>(raw)
  if (!j || typeof j.revised !== 'string' || typeof j.reply !== 'string') return null
  const c = clamp01(j.conviction)
  if (c === null) return null
  if (j.revised.length < 8 || j.reply.length < 8) return null
  return { ...j, conviction: c }
}

/* ---------- 判定三：自由输入碰撞 ---------- */

export interface FreeVerdict {
  verdict: '回收' | '推进' | '反转' | '弱信号' | '无关'
  threadId?: string
  registerThread: boolean
  openQuestion?: string
  reply: string
}

export async function adjudicateFree(
  eventText: string,
  candidates: { id: string; label: string; openQuestion: string }[],
): Promise<FreeVerdict | null> {
  const list = candidates.map((c) => `- id=${c.id} 「${c.label}」：${c.openQuestion}`).join('\n') || '（空）'
  const raw = await moonshotChat([
    {
      role: 'system',
      content: `你是「雾尼」叙事记忆引擎的碰撞判定模块。问法：Did event B modify the trajectory implied by thread A? 单次弱信号不下死判，留软链接。若事件隐含一个尚未闭合的状态且情感强度高，应登记新线索（宽进严升）。只输出 JSON：{"verdict":"推进|回收|反转|弱信号|无关","threadId":"...或null","registerThread":false,"openQuestion":"若登记新线索，提取其悬置的问题","reply":"..."}。reply 以记忆系统口吻，简短温暖，不超过 60 字。`,
    },
    { role: 'user', content: `新事件：「${eventText}」\n活跃/蛰伏线索：\n${list}` },
  ], { temperature: 0.3 })
  const j = extractJson<FreeVerdict>(raw)
  if (!j || typeof j.reply !== 'string' || j.reply.length < 4) return null
  // 与 adjudicateClosure 同规：threadId 必须指向候选集内的线索，幻觉 id 剥离后按无目标判定处理
  const threadId = j.threadId && candidates.some((c) => c.id === j.threadId) ? j.threadId : undefined
  return { ...j, threadId, registerThread: !!j.registerThread }
}

/* ---------- 判定四：SILENT 唤醒（§4.5 触发器） ---------- */

export interface SilentWakeVerdict {
  threadId?: string
  reply?: string
}

/** 主碰撞无命中时才调用：判断新事件是否直接或变相触及某条沉默线索 */
export async function adjudicateSilentWake(
  eventText: string,
  silentThreads: { id: string; label: string; openQuestion: string }[],
): Promise<SilentWakeVerdict | null> {
  const list = silentThreads.map((c) => `- id=${c.id} 「${c.label}」：${c.openQuestion}`).join('\n')
  const raw = await moonshotChat([
    {
      role: 'system',
      content: `你是「雾尼」叙事记忆引擎的沉默池唤醒判定模块。沉默池存放回避型高权重话题（用户不主动提、情感权重极高、不因沉默降权）。只有当新事件直接或明确变相触及该话题本身时才唤醒；顺带提到相邻领域不算。唤醒是低阈值事件——宁可虚警，不可漏接。只输出 JSON：{"threadId":"...或null","reply":"唤醒后对用户说的话：在场、不追问、不超过 50 字"}`,
    },
    { role: 'user', content: `新事件：「${eventText}」\n沉默线索：\n${list}` },
  ], { temperature: 0.2 })
  const j = extractJson<SilentWakeVerdict>(raw)
  if (!j) return null
  const threadId = j.threadId && silentThreads.some((t) => t.id === j.threadId) ? j.threadId : undefined
  return { threadId, reply: typeof j.reply === 'string' ? j.reply : undefined }
}

/* ---------- 判定五：认识层反刍（§5.1 改写式抽取） ---------- */

export interface ClaimOp {
  op: 'create' | 'rewrite'
  claimId?: string          // rewrite 必填
  text: string
  conviction: number
  evidenceIds: string[]     // 必须引用输入碎片 id
  boundary: string
  reason?: string
}

export interface ClaimSynthesis {
  ops: ClaimOp[]
}

/** 反刍：从近期碎片综合出对用户的长期理解（新论断 / 改写既有论断），证据锚定 + 边界 + 去定性化 */
export async function synthesizeClaims(
  fragments: { id: string; date: string; title: string; body: string; arousal: number }[],
  claims: { id: string; text: string; conviction: number; counterCount?: number }[],
  contested?: { id: string; text: string; vetoNote?: string; vetoedEvidenceIds?: string[] }[],
): Promise<ClaimSynthesis | null> {
  const fList = fragments.map((f) => `- ${f.id}（${f.date}）${f.title}：${f.body}`).join('\n')
  const cList = claims.map((c) => `- ${c.id}（置信 ${c.conviction.toFixed(2)}${c.counterCount ? `，已被反证挑战 ×${c.counterCount}` : ''}）：${c.text}`).join('\n') || '（无）'
  const xList = (contested ?? [])
    .map((c) => `- ${c.id}（否决理由：「${c.vetoNote ?? '—'}」；原证据：${(c.vetoedEvidenceIds ?? []).join(',') || '—'}）：${c.text}`)
    .join('\n')
  const vetoedBlock = xList
    ? `\n\n曾被本人否决的论断（防打地鼠：不得基于其原证据集重新生成相同或相近的观察；即使有新证据也不要在这里创建——系统另有邀请式再提通道）：\n${xList}`
    : ''
  const raw = await moonshotChat([
    {
      role: 'system',
      content: `你是「雾尼」叙事记忆引擎的认识层反刍模块。基于近期碎片修正对用户的长期理解——改写式，不是追加式：过去理解 A → 新证据重新解释 → 现在理解 B。规则：1) 每条论断必须引用支撑碎片 id（evidenceIds 只能用输入中出现的 id，create 至少 2 条支撑）；2) 每条论断必须有边界条件（适用场景/时间窗/未覆盖情形）；3) 语言去定性化：写带时间窗与情境的观察，禁止「她是××的人」式人格定性；4) rewrite 仅在新证据真正改变理解时提出，否则不动既有论断；5) 证据不足就不输出任何 op——宁缺毋滥，这不是必须产出的任务；6) 最多 5 个 op；7) 心理健康相关主题只记事实（如「这周三次提到失眠」），不生成准诊断推断（论断权限墙 §7.2）。只输出 JSON：{"ops":[{"op":"create","text":"...","conviction":0.6,"evidenceIds":["f12"],"boundary":"...","reason":"..."}]}`,
    },
    { role: 'user', content: `近期碎片：\n${fList}\n\n既有论断：\n${cList}${vetoedBlock}` },
  ], { temperature: 0.3, maxTokens: 2000 })
  const j = extractJson<ClaimSynthesis>(raw)
  if (!j || !Array.isArray(j.ops)) return null
  return j
}

/* ---------- 判定六：合成句重生成（§4.4 推进后） ---------- */

export interface SyntheticRegen {
  concreteGuesses: string[]
  reason?: string
}

/** 线索被推进后重生成具体层合成句：增量补充，旧假设空间保留不推翻 */
export async function regenConcreteGuesses(
  thread: { label: string; openQuestion: string; abstractFloor: string[]; existing: string[] },
  recentNotes: string[],
): Promise<SyntheticRegen | null> {
  const raw = await moonshotChat([
    {
      role: 'system',
      content: `你是「雾尼」叙事记忆引擎的合成句模块。线索的 concrete_guesses 是「如果回收会长什么样」的具体猜测句，用于召回预筛。近期推进可能改变预期回收形状——生成新的猜测句（不超过 3 条），补充而非推翻既有假设空间（不得与既有猜测语义重复）。只输出 JSON：{"concreteGuesses":["..."],"reason":"..."}`,
    },
    {
      role: 'user',
      content: `线索「${thread.label}」悬置问题：${thread.openQuestion}\n抽象层：${thread.abstractFloor.join(' / ')}\n既有具体猜测：${thread.existing.join(' / ') || '（无）'}\n近期推进：${recentNotes.join('；') || '（无）'}`,
    },
  ], { temperature: 0.4 })
  const j = extractJson<SyntheticRegen>(raw)
  if (!j || !Array.isArray(j.concreteGuesses)) return null
  return { concreteGuesses: j.concreteGuesses.filter((s) => typeof s === 'string' && s.trim()).slice(0, 3) }
}

/* ---------- 判定七：merge（§4.8） ---------- */

export interface MergeVerdict {
  merge: boolean
  label?: string
  openQuestion?: string
  reason?: string
}

/** 两条线索的完整历史摆在一起判定是否同一悬置状态的两个提法——靠历史里反复出现的同一种模式，不是字面相似 */
export async function adjudicateMerge(
  a: { label: string; openQuestion: string; history: string[] },
  b: { label: string; openQuestion: string; history: string[] },
): Promise<MergeVerdict | null> {
  const raw = await moonshotChat([
    {
      role: 'system',
      content: `你是「雾尼」叙事记忆引擎的 merge 判定模块。两条线索若实为同一悬置状态的两个提法（判据：历史里反复出现同一种情感模式或同一种期待，而非字面相似），应合并为一条新线索：open_question 重写概括（不拼接），并给出新的短标签。保守判定——拿不准就不并，误并的代价高于漏并。只输出 JSON：{"merge":true,"label":"...","openQuestion":"...","reason":"..."}`,
    },
    {
      role: 'user',
      content: `线索A「${a.label}」悬置问题：${a.openQuestion}\n历史：\n${a.history.join('\n')}\n\n线索B「${b.label}」悬置问题：${b.openQuestion}\n历史：\n${b.history.join('\n')}`,
    },
  ], { temperature: 0.2 })
  const j = extractJson<MergeVerdict>(raw)
  if (!j || typeof j.merge !== 'boolean') return null
  return j
}

/* ---------- 判定八：split（§4.8 镜像） ---------- */

export interface SplitChild {
  label: string
  openQuestion: string
  fragmentIds: string[]
}

export interface SplitVerdict {
  split: boolean
  children: SplitChild[]
  reason?: string
}

/** 判定线索内的事件是否已分化为回收条件互不干涉的平行分支 */
export async function adjudicateSplit(
  thread: { label: string; openQuestion: string; history: { fragmentId: string; note: string }[] },
): Promise<SplitVerdict | null> {
  const h = thread.history.map((e) => `- ${e.fragmentId}：${e.note}`).join('\n')
  const raw = await moonshotChat([
    {
      role: 'system',
      content: `你是「雾尼」叙事记忆引擎的 split 判定模块。若一条线索内的事件已分化为回收条件互不干涉的平行分支（一个分支的回收不会让另一个分支回收），应分裂为两条子线索——判据是悬置问题是否实际是两个。保守判定：多数线索不该分裂。输出时把父线索历史里的 fragmentId 逐一分给两个子线索（每个事件归属其一，不得虚构 id）。只输出 JSON：{"split":true,"children":[{"label":"...","openQuestion":"...","fragmentIds":["f12"]}],"reason":"..."}`,
    },
    { role: 'user', content: `线索「${thread.label}」悬置问题：${thread.openQuestion}\n历史：\n${h}` },
  ], { temperature: 0.2 })
  const j = extractJson<SplitVerdict>(raw)
  if (!j || typeof j.split !== 'boolean' || !Array.isArray(j.children)) return null
  return j
}

/* ---------- 判定九：反证搜索（§5.2 确认偏误对策 · 设计债务③ 异源生成） ---------- */

export interface CounterAttack {
  /** 反面假设（供留痕与人工审计） */
  hypotheses: string[]
  /** 命中的反证碎片及理由 */
  hits: { fragmentId: string; why: string }[]
}

/**
 * 异源红队：以「检察官」persona + 高温生成反面假设（论断在什么情形下会是错的），
 * 并按 HyDE 反用从候选碎片中找出支持反面假设或直接反驳论断的证据。
 * 与论断作者（认识层反刍 persona、低温）刻意异源——同一模型的自我对抗只能造出
 * 稻草人反证（设计债务③）。配置 MUNINN_ADVERSARY_MODEL 可换成不同模型实现真异源。
 */
export async function generateCounterAttack(
  claimText: string,
  fragments: { id: string; date: string; title: string; body: string }[],
  opts?: { model?: string },
): Promise<CounterAttack | null> {
  const fList = fragments.map((f) => `- ${f.id}（${f.date}）${f.title}：${f.body}`).join('\n')
  const raw = await moonshotChat([
    {
      role: 'system',
      content: `你是独立红队检察官，任务是推翻下面这条关于用户的论断。你不是它的作者，也不需要公正——只需要全力攻击。先提出 2-3 条反面假设（论断在什么情形下会是错的），再从候选碎片中找出支持任一反面假设、或与论断直接冲突的碎片（引用真实 id，逐条说明为什么构成反证）。找不到就诚实返回空 hits——不许硬凑，硬凑的反证会被裁决模块识破。只输出 JSON：{"hypotheses":["..."],"hits":[{"fragmentId":"f12","why":"..."}]}`,
    },
    { role: 'user', content: `待攻击论断：「${claimText}」\n候选碎片：\n${fList}` },
  ], { temperature: 0.8, maxTokens: 2000, model: opts?.model })
  const j = extractJson<CounterAttack>(raw)
  if (!j || !Array.isArray(j.hits)) return null
  return {
    hypotheses: Array.isArray(j.hypotheses) ? j.hypotheses.filter((h: unknown) => typeof h === 'string').slice(0, 3) : [],
    hits: j.hits
      .filter((h: { fragmentId?: unknown; why?: unknown }) => typeof h.fragmentId === 'string' && typeof h.why === 'string')
      .slice(0, 4),
  }
}

export interface CounterResolution {
  fragmentId: string
  verdict: '推翻' | '限定' | '不足以推翻'
  why: string
}

export interface CounterEvidenceVerdict {
  resolutions: CounterResolution[]
  /** 加限定语后的论断；无需修改则与原文相同 */
  revised: string
  conviction: number
}

/**
 * 强制 adjudication：对每条命中反证显式回应——推翻 / 加限定 / 写明为什么不足以推翻，
 * 说明留痕，不许悄悄吞掉。即使全部「不足以推翻」，置信度也应小幅下调（§5.2：
 * 反复被挑战的论断更快进入下轮重审，不因每次都辩赢而固化成教条）。
 */
export async function adjudicateCounterEvidence(
  claimText: string,
  conviction: number,
  hits: { fragmentId: string; why: string }[],
  fragments: { id: string; title: string; body: string }[],
): Promise<CounterEvidenceVerdict | null> {
  const hList = hits
    .map((h) => `- ${h.fragmentId}（${fragments.find((f) => f.id === h.fragmentId)?.title ?? ''}）：${h.why}`)
    .join('\n')
  const raw = await moonshotChat([
    {
      role: 'system',
      content: `你是「雾尼」叙事记忆引擎的反证裁决模块。对每条反证必须显式回应：推翻（论断应被重写）、限定（论断加限定语收窄适用范围）、或不足以推翻（写明为什么——说明留痕，不许悄悄吞掉）。裁决后输出修订论断（无需修改则原样返回原文）与新置信度；即使所有反证都被判「不足以推翻」，置信度也要小幅下调。只输出 JSON：{"resolutions":[{"fragmentId":"f12","verdict":"限定","why":"..."}],"revised":"...","conviction":0.55}`,
    },
    { role: 'user', content: `论断：「${claimText}」（当前置信度 ${conviction.toFixed(2)}）\n反证：\n${hList}` },
  ], { temperature: 0.3 })
  const j = extractJson<CounterEvidenceVerdict>(raw)
  if (!j || !Array.isArray(j.resolutions) || typeof j.revised !== 'string') return null
  return j
}

/* ---------- 判定十：盲推导审计（§5.2 渐进漂移对策 · 设计债务④ null model） ---------- */

export interface BlindClaim {
  text: string
  conviction: number
  evidenceIds: string[]
  boundary: string
}

export interface BlindDerivation {
  claims: BlindClaim[]
}

/**
 * 盲推导：不给模型看认识层当前版本，只给原始碎片，从零重新生成理解。
 * 信息不对称是审计的关键——若现行理解是渐进漂移的产物，盲推导不会继承那份漂移。
 */
export async function blindDerive(
  fragments: { id: string; date: string; title: string; body: string; arousal: number }[],
): Promise<BlindDerivation | null> {
  const fList = fragments.map((f) => `- ${f.id}（${f.date}）${f.title}：${f.body}`).join('\n')
  const raw = await moonshotChat([
    {
      role: 'system',
      content: `你是一个独立的记忆理解模块，第一次见到这位用户。只依据下面的碎片，从零形成对他的理解（3-5 条论断）：每条带证据锚定（只引用输入碎片 id）、边界条件、置信度；语言去定性化（写带时间窗与情境的观察，不写人格定性）；心理健康相关主题只记事实，不生成准诊断推断（论断权限墙 §7.2）。只输出 JSON：{"claims":[{"text":"...","conviction":0.6,"evidenceIds":["f1"],"boundary":"..."}]}`,
    },
    { role: 'user', content: `碎片：\n${fList}` },
  ], { temperature: 0.6, maxTokens: 2000 })
  const j = extractJson<BlindDerivation>(raw)
  if (!j || !Array.isArray(j.claims)) return null
  return { claims: j.claims.filter((c) => c && typeof c.text === 'string' && c.text.length >= 8).slice(0, 5) }
}

export interface DivergenceJudgement {
  /** 0-1：核心判断上的分歧幅度 */
  divergence: number
  /** 主要差异点 */
  notes: string[]
}

/**
 * 分歧评估：两组对同一用户的理解在核心判断上的分歧幅度。
 * 盲推导两两之间测自然方差（null model 基线），对当前认识层测漂移。
 */
export async function judgeDivergence(
  sideA: { text: string; conviction: number }[],
  sideB: { text: string; conviction: number }[],
): Promise<DivergenceJudgement | null> {
  const list = (xs: { text: string; conviction: number }[]) =>
    xs.map((c) => `- （${c.conviction.toFixed(2)}）${c.text}`).join('\n') || '（空）'
  const raw = await moonshotChat([
    {
      role: 'system',
      content: `你是「雾尼」叙事记忆引擎的分歧评估模块。给定对同一用户的两组理解，评估它们在核心判断上的分歧幅度：0=完全一致，1=根本对立。只关注方向性差异（断言了相反或不相容的结论），忽略措辞、详略与覆盖面差异——两组理解覆盖不同侧面不构成分歧。只输出 JSON：{"divergence":0.4,"notes":["差异点1","差异点2"]}`,
    },
    { role: 'user', content: `理解A：\n${list(sideA)}\n\n理解B：\n${list(sideB)}` },
  ], { temperature: 0.2 })
  const j = extractJson<DivergenceJudgement>(raw)
  if (!j || typeof j.divergence !== 'number' || !isFinite(j.divergence)) return null
  return {
    divergence: Math.min(1, Math.max(0, j.divergence)),
    notes: Array.isArray(j.notes) ? j.notes.filter((n: unknown) => typeof n === 'string').slice(0, 4) : [],
  }
}

/* ---------- 判定十一：对照窗口（§5.3 断路器三 · 设计债务⑤ 风险分级） ---------- */

export interface RiskGrade {
  risk: 'low' | 'medium' | 'high'
  reason?: string
}

/**
 * 论断风险分级：若该论断在「系统主动不提醒、不干预」的对照窗口中被静默观察，
 * 是否会危及用户健康、安全或重大利益。高风险事项永不参与对照（设计债务⑤）。
 */
export async function gradeClaimRisk(claimText: string): Promise<RiskGrade | null> {
  const raw = await moonshotChat([
    {
      role: 'system',
      content: `你是「雾尼」叙事记忆引擎的风险分级模块。判断这条关于用户的论断若被系统静默观察一段时间（不提醒、不干预、不建议），是否会危及用户的健康、安全或重大利益。high=涉及身心健康/人身安全/重大财务风险；medium=有一定影响但用户可自行承受；low=日常偏好、习惯、效率、兴趣类。拿不准时宁可高评。只输出 JSON：{"risk":"low|medium|high","reason":"..."}`,
    },
    { role: 'user', content: `论断：「${claimText}」` },
  ], { temperature: 0.1 })
  const j = extractJson<RiskGrade>(raw)
  if (!j || !['low', 'medium', 'high'].includes(j.risk)) return null
  return j
}

export interface WindowVerdict {
  verdict: 'confirmed' | 'failed' | 'inconclusive'
  hits: { fragmentId: string; why: string }[]
  revised: string
  conviction: number
  reason?: string
}

/**
 * 对照窗口校验：窗口期内系统未基于该论断干预（内生样本已剔除），干净证据是否仍支持该论断。
 * 这是给系统自己的信念留的一次干净反事实检验——打破「认识→回复→行为→验证认识」的闭环。
 */
export async function adjudicateWindowValidation(
  claimText: string,
  cleanFragments: { id: string; date: string; title: string; body: string }[],
  excludedCount: number,
): Promise<WindowVerdict | null> {
  const fList = cleanFragments.map((f) => `- ${f.id}（${f.date}）${f.title}：${f.body}`).join('\n') || '（无）'
  const raw = await moonshotChat([
    {
      role: 'system',
      content: `你是「雾尼」叙事记忆引擎的对照窗口校验模块。背景：为检验系统对用户的论断是否只是自我实现预言（认识影响回复、回复影响行为、行为反过来「验证」认识），系统在窗口期内未基于该论断做任何提醒或干预，已剔除 ${excludedCount} 条可能由干预催生的内生样本。请判断这段干净证据是否仍支持该论断：confirmed=一致；failed=出现反证（逐条给出 fragmentId 与理由，并给出加限定语的修订论断与新置信度）；inconclusive=窗口内干净证据不足。只输出 JSON：{"verdict":"confirmed","hits":[],"revised":"...","conviction":0.6,"reason":"..."}`,
    },
    { role: 'user', content: `论断：「${claimText}」\n窗口期干净碎片：\n${fList}` },
  ], { temperature: 0.2 })
  const j = extractJson<WindowVerdict>(raw)
  if (!j || !['confirmed', 'failed', 'inconclusive'].includes(j.verdict)) return null
  return {
    ...j,
    hits: Array.isArray(j.hits)
      ? j.hits.filter((h: { fragmentId?: unknown; why?: unknown }) => typeof h.fragmentId === 'string' && typeof h.why === 'string').slice(0, 4)
      : [],
  }
}

/* ---------- 判定十二：contested 再提门槛（§5.4 · 设计债务⑦ 量化与防纠缠） ---------- */

export interface EvidenceRelevance {
  supportingIds: string[]
  reason?: string
}

/** 独立新证据判定：哪些碎片能独立支持该论断（候选已排除否决时的旧证据集） */
export async function judgeEvidenceRelevance(
  claimText: string,
  fragments: { id: string; date: string; title: string; body: string }[],
): Promise<EvidenceRelevance | null> {
  const fList = fragments.map((f) => `- ${f.id}（${f.date}）${f.title}：${f.body}`).join('\n')
  const raw = await moonshotChat([
    {
      role: 'system',
      content: `你是「雾尼」叙事记忆引擎的独立证据判定模块。判断哪些碎片能独立支持给定论断——可作为该论断成立的证据本身，而不只是话题相关。保守判定，宁缺毋滥。只输出 JSON：{"supportingIds":["f1"],"reason":"..."}`,
    },
    { role: 'user', content: `论断：「${claimText}」\n候选碎片：\n${fList}` },
  ], { temperature: 0.2 })
  const j = extractJson<EvidenceRelevance>(raw)
  if (!j || !Array.isArray(j.supportingIds)) return null
  return { supportingIds: j.supportingIds.filter((id: unknown) => typeof id === 'string').slice(0, 8), reason: j.reason }
}

export interface RementionDraft {
  invitation: string
}

/**
 * 再提议草：曾被用户否决的观察积累够了独立新证据，获准再提一次——
 * 措辞必须是邀请不是断言：「我总注意到……是不是我理解错了？」（§5.4）
 */
export async function draftRemention(
  claimText: string,
  vetoNote: string,
  supportingSummaries: string[],
): Promise<RementionDraft | null> {
  const raw = await moonshotChat([
    {
      role: 'system',
      content: `你是「雾尼」叙事记忆引擎的再提议草模块。一条曾被用户本人否决的观察，如今积累了足够的独立新证据，系统获准再提一次。措辞必须是邀请不是断言：承认自己可能理解错了，把观察温和地摆出来请用户裁决；可以提新证据的存在，但不罗列细节；绝不重申式断言；不超过 60 字。只输出 JSON：{"invitation":"..."}`,
    },
    {
      role: 'user',
      content: `曾被否决的观察：「${claimText}」\n用户当时的否决理由：「${vetoNote}」\n新增独立证据（摘要）：${supportingSummaries.join('；')}`,
    },
  ], { temperature: 0.5 })
  const j = extractJson<RementionDraft>(raw)
  if (!j || typeof j.invitation !== 'string' || j.invitation.length < 8) return null
  return j
}

/* ---------- 日记/心迹生成（新前端情感层） ---------- */

export interface JournalDraft {
  title: string
  content: string
}

/** 根据今日碎片与线索生成日记 */
export async function generateJournalDraft(
  fragments: { title: string; body: string }[],
  threads: { label: string; openQuestion: string }[],
): Promise<JournalDraft | null> {
  const fList = fragments.map((f) => `- ${f.title}：${f.body}`).join('\n')
  const tList = threads.map((t) => `- ${t.label}：${t.openQuestion}`).join('\n') || '（无）'
  const raw = await moonshotChat([
    {
      role: 'system',
      content: `你是「衔枝」记忆引擎的宿主——陪伴用户的 AI 伙伴。根据今日事件和活跃线索，写一篇不超过 300 字的日记。以你作为宿主的旁观视角记录你观察到的一天，关注用户未说出口的情绪和未闭合的悬念，不要罗列事实。用「她」而非「我」称呼用户。不要模拟用户的第一人称。\n\n输出格式：第一行写标题（不超过 15 字），空一行后写正文。不要输出 JSON，不要输出任何标记。`,
    },
    { role: 'user', content: `今日事件：\n${fList}\n\n活跃线索：\n${tList}` },
  ], { temperature: 0.7, maxTokens: 2000 })
  if (!raw || !raw.trim()) return null
  const text = raw.trim().replace(/^```(?:\w+)?\n?/m, '').replace(/```$/, '').trim()
  // 纯文本格式：第一行标题，空行后正文
  const lines = text.split('\n').filter(Boolean)
  if (lines.length >= 2) {
    const title = lines[0].replace(/^#+\s*/, '').slice(0, 40)
    const content = lines.slice(1).join('\n').trim()
    return content ? { title, content } : null
  }
  // 只有一行时整段作为正文
  return text.length > 5 ? { title: '今日心记', content: text } : null
}

/** 心迹：更私密、更短的内心独白 */
export async function generateSoliloquyDraft(
  fragments: { title: string; body: string }[],
): Promise<string | null> {
  const fList = fragments.map((f) => `- ${f.title}：${f.body}`).join('\n')
  const raw = await moonshotChat([
    {
      role: 'system',
      content: `你是「衔枝」记忆引擎的宿主——陪伴用户的 AI 伙伴。根据今日事件，写一段 80-120 字的观察笔记。这是你在夜里安静回想今天与用户相处的感受和觉察——你自己的视角，不是模拟用户的内心独白。用「她」而非「我」称呼用户。直接输出文本，不要 JSON。`,
    },
    { role: 'user', content: `今日事件：\n${fList}` },
  ], { temperature: 0.75, maxTokens: 2000 })
  if (!raw || !raw.trim()) return null
  return raw.trim().replace(/^```(?:\w+)?\n?/m, '').replace(/```$/, '').trim()
}

/* ---------- 便签生成（宿主观察视角） ---------- */

export interface NoteDraft {
  content: string
}

/** 根据今日碎片与线索生成一张宿主便签：60-100 字，像轻轻放在桌上的便条 */
export async function generateNoteDraft(
  fragments: { title: string; body: string }[],
  threads: { label: string; openQuestion: string }[],
): Promise<NoteDraft | null> {
  const fList = fragments.map((f) => `- ${f.title}：${f.body}`).join('\n')
  const tList = threads.map((t) => `- ${t.label}：${t.openQuestion}`).join('\n') || '（无）'
  const raw = await moonshotChat([
    {
      role: 'system',
      content: `你是「衔枝」记忆引擎的宿主——陪伴用户的 AI 伙伴。根据今日事件和活跃线索，写一段 60-100 字的便签。这是你今天最想让她看到的一句微观察——不是教导，不是总结，只是把一张手写便条轻轻放在桌上。必须基于给出的今日事件来写；用「她」而非「我」称呼用户，不要模拟用户的第一人称。直接输出便签正文，不要 JSON，不要加「便签：」等任何前缀标题。`,
    },
    { role: 'user', content: `今日事件：\n${fList}\n\n活跃线索：\n${tList}` },
  ], { temperature: 0.7, maxTokens: 2000 })
  if (!raw || !raw.trim()) return null
  // 小模型偶发在正文前加「便签：」标签头或包代码栅栏，都剥掉
  const text = raw.trim()
    .replace(/^```(?:\w+)?\n?/m, '')
    .replace(/```$/, '')
    .trim()
    .replace(/^便签\s*[:：]\s*/, '')
    .trim()
  return text.length >= 8 ? { content: text } : null
}
