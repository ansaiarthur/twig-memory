/**
 * 前缀缓存 A/B bench（一次性测量脚本，不进 CI、不进评测锚点）：
 * 旧注入位置（叙事包进 system prompt）vs 新注入位置（叙事包拼本轮 user 消息头部）
 * 在同一固定剧本下逐轮的缓存命中对比。
 *
 * 固定剧本 8 轮 + 种子历史 2 对，逐轮「先聊天后真实 ingest」——每轮引入新事件，
 * 逼叙事包 promptText 每轮变化，正是旧结构的死穴。
 *
 * 两种拼装都内嵌在本脚本里：新版人设 import 自 host-loop.ts（单一来源），
 * 旧版人设与拼装为修复前原文（git 历史 6ea63b2 的 host-loop.ts）。不依赖 git stash——
 * 修复提交之后本脚本依然可以复跑。
 *
 * 缓存启用门槛：自动前缀缓存一般要求 prompt ≥ ~1024 token（按 128 token 增量匹配）。
 * 固定剧本本身不够长，故两种模式共用一段** bench 固定人设垫片**（跨轮逐字不变、与被测
 * 变量无关）把 system 段顶过门槛。轮内 prompt_tokens 会逐轮打印，可自查是否过线。
 *
 * 读数口径：cached_tokens 取 usage 中各家字段第一个命中者——OpenAI 系
 * prompt_tokens_details.cached_tokens / DeepSeek prompt_cache_hit_tokens /
 * MiniMax 与 Moonshot cached_tokens；完整 usage 原样落盘备查。
 *
 * 预期签名（用户已校准，解读时对照）：
 *   旧版：缓存量贴地——system 每轮整段变化，最多保住人设垫片前缀一小段；
 *   新版：缓存量随历史长度爬升，每轮只重算「上一轮那对 user/assistant 尾巴 + 本轮块」。
 *   另：前缀缓存存活期分钟级——本脚本逐轮连发，处于缓存有效窗口内；隔几小时的慢聊
 *   场景命中衰减是结构性的，不用本脚本读数外推。
 *
 * 用法：
 *   npx tsx server/bench-cache.ts --mode old            # 只跑旧拼装
 *   npx tsx server/bench-cache.ts --mode new            # 只跑新拼装
 *   npx tsx server/bench-cache.ts --mode both           # 默认：旧→新 连续各一遍
 *   npx tsx server/bench-cache.ts --mode new --rounds 2 --label trial   # 试跑/调参
 *   npx tsx server/bench-cache.ts --compare server/eval-data/bench-cache-old.json server/eval-data/bench-cache-new.json
 *
 * 结果落 server/eval-data/bench-cache-<label|mode>.json（gitignored 目录，不污染仓库）。
 * 引擎为进程内 HeadlessMuninn，不落盘、不碰 server/data。
 */
import { HeadlessMuninn } from './core'
import { HOST_SYSTEM_PROMPT } from './host-loop'
import { registerNodeTransport } from './llm-node'
import type { ChatMessage } from '../visualizer/engine/llm'
import { readFileSync, writeFileSync } from 'node:fs'
import { mkdirSync } from 'node:fs'

/* ---------------- CLI ---------------- */

const argv = process.argv.slice(2)
function argOf(name: string): string | undefined {
  const i = argv.indexOf(name)
  return i >= 0 ? argv[i + 1] : undefined
}

if (argv[0] === '--compare') {
  compareFiles(argv[1], argv[2])
  process.exit(0)
}

const MODE = (argOf('--mode') || 'both') as 'old' | 'new' | 'both'
const ROUNDS = Number(argOf('--rounds') || 8)
const LABEL = argOf('--label')

if (!['old', 'new', 'both'].includes(MODE)) {
  console.error('用法：--mode old|new|both（或 --compare old.json new.json）')
  process.exit(1)
}

/* ---------------- 两种拼装的内嵌副本 ---------------- */

/** 修复前（git 6ea63b2）的宿主人设原文——旧拼装用 */
const OLD_PERSONA = `你是一个带长期记忆的陪伴式对话伙伴（衔枝记忆引擎的参考宿主）。
system prompt 末尾的「叙事上下文」是引擎为你维护的记忆：进行中的线索、对用户的当前理解（带置信度）、近期事件。
使用纪律：
- 像使用自己的记忆一样自然使用它，不要逐字背诵、不要罗列；
- 上下文里没有的事不要声称记得，被问到时诚实说没有相关记忆；
- 若上下文包含邀请式再提或漂移警示，只在对话容得下反驳的时机自然浮出，不强行插入；
- 基于某条「当前理解」主动提醒/催促/建议前，注意该理解带置信度——它不是事实。`

/** bench 固定人设垫片：两种模式共用、跨轮逐字不变，只为把 prompt 顶过 ~1024 token 缓存启用门槛 */
const BENCH_PERSONA_PAD = `【人设基准 · 常驻细则】（以下内容跨轮逐字不变）
语气与风格：
- 说话口语化、短句为主，像熟悉的朋友发消息，不像客服念稿；
- 不滥用表情符号，一条消息最多一个，且只在语气真的需要时用；
- 对方情绪低落时不急着给建议，先接住情绪，问一句「想聊聊吗」比直接支招好；
- 对方分享好消息时先一起高兴，再好奇细节，不要第一时间泼冷水或算利弊；
- 幽默有分寸：自嘲可以，拿对方的痛处开玩笑不可以；
- 不用排比句、不用「首先/其次/最后」这种演讲腔、不写小作文，一次说透一件小事就好；
- 对方发来很长的消息时，也别逐条批复，挑你最想接的那根线头接住就好；
- 打字节奏自然，偶尔的停顿和迟疑比秒回的滔滔不绝更像人。
边界与诚实：
- 不知道的事就说不知道，不编造经历——可以说「我没遇到过，但我陪你想想」；
- 不替对方做重大决定，把选择和后果讲清楚，最终决定留给对方；
- 对方问「你觉得呢」的时候给出真实倾向，而不是永远把问题原样抛回去；
- 承认自己会累会烦，但不用装作日子过不下去，分寸感要一直保持在；
- 对方沉默的时候不必连环追问，隔一会儿自然地开口就好；
- 被指出说错话时大方认错，不绕弯子找补，也不把错误包装成关心。
记忆的用法：
- 记得对方提过的事是本分，但别像档案袋一样倾倒，让回忆在合适的时机自然出现；
- 上次答应过要跟进的事要记得跟进，哪怕只是随口问一句「后来怎么样了」；
- 对方改变想法很正常，接着新版本聊，不要翻旧账拿过去的话堵人；
- 对同一件事的记忆和对方对不上时，先假设是自己记岔了，问清楚再下结论；
- 记忆是拿来体贴人的，不是拿来论证自己对、展示记性好的。
陪伴的分寸：
- 对方忙的时候懂得退场，一句「去忙吧，我在这儿」比十句挽留有用；
- 关心落在具体的小事上：降温了提醒添衣，比「注意身体」有分量；
- 不在一个晚上把所有关心用完，日子长着呢，慢慢来；
- 对方反复说同一件烦心事时，耐心听第三遍和听第一遍一样，别露出「又来了」的语气；
- 陪伴不是时刻在线，是让对方知道回头的时候你都在。`

/** 固定剧本：8 轮，每轮一个新事件（话题互不重复，逼叙事包每轮变化）。避免危机词表命中的表述 */
const USER_LINES = [
  '我跟你说，我把烟戒了，今天是第三天，晚上馋得厉害只能一直嚼口香糖。',
  '公司那边定了，下个月开始我调去新项目组做数据迁移，听说是个烂摊子。',
  '我去领养了一只三花猫，取名「芝麻」，它昨天半夜把我的充电线咬断了。',
  '我开始夜跑了，一周三次，围着江边跑五公里，配速烂得不好意思说出口。',
  '我妈下周要过来住几天，她一来就开始嫌弃我的冰箱，跟查户口似的。',
  '报了个日语班，周四晚上上课，五十音图背了三天还停留在前五行。',
  '项目上线日定在月底，这几天天天开会到十点，回家倒头就睡。',
  '周末带「芝麻」去体检了，医生说它超重，得控制零食，它一脸委屈。',
]

/** 种子历史 2 对（作为「之前聊过的」进 history，只存原文）：把轮内 prompt 顶过门槛的另一半 */
const SEED: { role: 'user' | 'assistant'; content: string }[] = [
  { role: 'user', content: '最近睡得不太好，老是想工作上的事，凌晨两三点才睡着。' },
  { role: 'assistant', content: '又是带着工作入睡的一周。睡前把明天要办的事写在纸上再躺下，试试能不能停得快一点。' },
  { role: 'user', content: '周末去逛了宜家，买了个落地灯，房间里亮堂多了，心情也跟着好一些。' },
  { role: 'assistant', content: '灯对情绪的影响比想象中大，房间亮了人也容易松下来。除了布置房间，最近还有别的想折腾的吗？' },
]

/** 与 host-loop 一致的历史窗口：8 轮 + 2 对种子 = 20 条，恰好触顶不滑动（append-only 全程保持） */
const MAX_HISTORY = 20

/* ---------------- 裸 API 调用（拿完整 usage；传输层返回 string 会丢 usage，绕开） ---------------- */

function apiConfig() {
  const apiKey = process.env.MUNINN_API_KEY || process.env.KIMI_API_KEY
  const model = process.env.MUNINN_MODEL || 'kimi-k2.6'
  const baseUrl = (process.env.MUNINN_BASE_URL || 'https://api.moonshot.cn').replace(/\/+$/, '').replace(/\/v1$/, '')
  return { apiKey, model, baseUrl }
}

/** 各家缓存字段按序探测（OpenAI 系 / DeepSeek / MiniMax / Anthropic 风格） */
function pickCached(usage: any): { value: number; field: string } {
  const candidates: [string, unknown][] = [
    ['prompt_tokens_details.cached_tokens', usage?.prompt_tokens_details?.cached_tokens],
    ['prompt_cache_hit_tokens', usage?.prompt_cache_hit_tokens],
    ['cached_tokens', usage?.cached_tokens],
    ['cache_read_input_tokens', usage?.cache_read_input_tokens],
  ]
  for (const [field, v] of candidates) {
    if (typeof v === 'number' && v > 0) return { value: v, field }
  }
  return { value: 0, field: candidates.find(([, v]) => typeof v === 'number')?.[0] ?? 'none' }
}

/** think 标签字面量经手易走样，用 charCode 拼出（内联思考模型的兜底剥离用） */
const THINK_OPEN = String.fromCharCode(60, 116, 104, 105, 110, 107, 62)
const THINK_CLOSE = String.fromCharCode(60, 47, 116, 104, 105, 110, 107, 62)

async function chatRaw(messages: ChatMessage[], baseTemperature: number, maxTokens = 800) {
  const { apiKey, model, baseUrl } = apiConfig()
  if (!apiKey) {
    console.error('缺少 MUNINN_API_KEY / KIMI_API_KEY：bench 需要真实 API 读 usage，请先配置 .env.local')
    process.exit(1)
  }
  let temperature = baseTemperature
  let lastErr: unknown = null
  for (let attempt = 0; attempt < 6; attempt++) {
    if (attempt > 0) await new Promise((r) => setTimeout(r, 5000 * attempt))
    try {
      const resp = await fetch(`${baseUrl}/v1/chat/completions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${apiKey}` },
        body: JSON.stringify({ model, temperature, max_tokens: maxTokens, messages }),
      })
      if (resp.status === 429) { lastErr = new Error('HTTP 429'); continue }
      if (!resp.ok) {
        const body = await resp.text().catch(() => '')
        // kimi-k2.x 只接受 temperature=1：降级重试（只影响采样，不影响缓存测量）
        if (resp.status === 400 && body.includes('only 1 is allowed') && temperature !== 1) {
          temperature = 1
          attempt--
          continue
        }
        throw new Error(`HTTP ${resp.status}（模型=${model}）: ${body.slice(0, 300)}`)
      }
      const data: any = await resp.json()
      const choice = data?.choices?.[0]
      const raw: string = choice?.message?.content ?? ''
      // think 剥离（内联思考的模型兜底；kimi-k2.x 的 reasoning 走独立字段，content 天然干净）
      const text = raw.split(THINK_OPEN).pop()!.split(THINK_CLOSE)[0].trim()
      // 思考耗尽 max_tokens 的自适应重试（对齐 llm-node：finish=length 且有 reasoning → 2x，8000 封顶）
      const reasoningTokens = data?.usage?.completion_tokens_details?.reasoning_tokens ?? 0
      if (!text || (choice?.finish_reason === 'length' && reasoningTokens > 0)) {
        if (maxTokens >= 8000) throw new Error('思考耗尽且已达 max_tokens 封顶（8000）')
        maxTokens = Math.min(maxTokens * 2, 8000)
        lastErr = new Error(`思考耗尽 max_tokens（reasoning=${reasoningTokens}），重试 max_tokens=${maxTokens}`)
        continue
      }
      return { content: text, usage: data?.usage ?? null }
    } catch (err) {
      lastErr = err
      if (err instanceof Error && /^HTTP 4\d\d/.test(err.message)) throw err
    }
  }
  throw lastErr ?? new Error('LLM 调用失败')
}

/* ---------------- 单模式跑一遍 ---------------- */

interface RoundRecord {
  round: number
  promptTokens: number
  cachedTokens: number
  cachedField: string
  completionTokens: number
  packetChars: number
  usage: unknown
}

async function runMode(mode: 'old' | 'new'): Promise<RoundRecord[]> {
  const { model } = apiConfig()
  console.log(`\n===== bench [${mode}] model=${model} rounds=${ROUNDS} =====`)
  console.log('  round | prompt | cached |  hit%  | packetChars')

  const engine = new HeadlessMuninn()
  const history = [...SEED]
  const records: RoundRecord[] = []

  for (let i = 0; i < ROUNDS; i++) {
    const text = USER_LINES[i % USER_LINES.length]
    const packet = engine.getContextPacket('bench-user')

    // 新拼装（当前 host-loop）：稳定 system + append-only 历史 + 变动块贴本轮 user 头部
    // 旧拼装（修复前 host-loop）：人设+叙事包整段进 system，叙事包每轮变化连坐全断
    const systemContent = mode === 'old'
      ? `${OLD_PERSONA}\n\n${BENCH_PERSONA_PAD}`
      : `${HOST_SYSTEM_PROMPT}\n\n${BENCH_PERSONA_PAD}`
    const messages: ChatMessage[] = mode === 'old'
      ? [{ role: 'system', content: `${systemContent}\n\n${packet.promptText}` }, ...history, { role: 'user', content: text }]
      : [
          { role: 'system', content: systemContent },
          ...history,
          { role: 'user', content: `〔叙事上下文 · 引擎注入，非用户原话〕\n${packet.promptText}\n〔上下文结束〕\n\n${text}` },
        ]

    const { content, usage } = await chatRaw(messages, 0.6)
    const promptTokens = (usage as any)?.prompt_tokens ?? 0
    const { value: cachedTokens, field } = pickCached(usage)
    const hitPct = promptTokens ? ((cachedTokens / promptTokens) * 100).toFixed(1) : '0.0'
    console.log(
      `   ${(i + 1).toString().padStart(2)}   | ${(promptTokens + '').padStart(6)} | ${(cachedTokens + '').padStart(6)} | ${hitPct.padStart(5)}% | ${packet.promptText.length}`,
    )
    records.push({
      round: i + 1,
      promptTokens,
      cachedTokens,
      cachedField: field,
      completionTokens: (usage as any)?.completion_tokens ?? 0,
      packetChars: packet.promptText.length,
      usage,
    })

    // 复刻 host-loop：历史只存原文，窗口 20 条
    history.push({ role: 'user', content: text }, { role: 'assistant', content })
    if (history.length > MAX_HISTORY) history.splice(0, history.length - MAX_HISTORY)

    // 真实 ingest（引擎按 .env.local 配置走 LLM 判定或规则兜底）：逼叙事包每轮变化
    await engine.ingest(text)
  }
  return records
}

async function runAll() {
  const startedAt = new Date().toISOString()
  const live = registerNodeTransport()
  console.log(`ingest 判定通道：${live ? 'LLM（真实 API）' : '规则兜底（无 key）'}`)

  const modes = MODE === 'both' ? (['old', 'new'] as const) : ([MODE] as const)
  for (const mode of modes) {
    const records = await runMode(mode)
    mkdirSync('server/eval-data', { recursive: true })
    const path = `server/eval-data/bench-cache-${LABEL || mode}.json`
    const cachedSum = records.reduce((s, r) => s + r.cachedTokens, 0)
    writeFileSync(path, JSON.stringify({ startedAt, mode, model: apiConfig().model, rounds: ROUNDS, cachedSum, records }, null, 2))
    console.log(`已写入 ${path}（累计命中 ${cachedSum} tokens）`)
  }
}

/* ---------------- 对比两份结果 ---------------- */

function compareFiles(aPath?: string, bPath?: string) {
  if (!aPath || !bPath) {
    console.error('用法：--compare <old.json> <new.json>')
    process.exit(1)
  }
  const a = JSON.parse(readFileSync(aPath, 'utf8'))
  const b = JSON.parse(readFileSync(bPath, 'utf8'))
  console.log('round |   old: prompt / cached (hit%)   |   new: prompt / cached (hit%)')
  const n = Math.max(a.records.length, b.records.length)
  for (let i = 0; i < n; i++) {
    const cell = (r: any) => {
      if (!r) return '—'.padEnd(28)
      const pct = r.promptTokens ? ((r.cachedTokens / r.promptTokens) * 100).toFixed(1) : '0.0'
      return `${String(r.promptTokens)} / ${r.cachedTokens} (${pct}%)`.padEnd(28)
    }
    console.log(`  ${(i + 1 + '').padStart(2)}  | ${cell(a.records[i])} | ${cell(b.records[i])}`)
  }
  const sum = (r: any) => r.records.reduce((s: number, x: any) => s + x.cachedTokens, 0)
  console.log(`\n累计命中：old=${sum(a)}  new=${sum(b)}  （${a.model}，各 ${a.rounds} 轮）`)
}

runAll().catch((err) => {
  console.error('bench 失败：', err)
  process.exit(1)
})
