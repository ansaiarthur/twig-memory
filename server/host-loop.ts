/**
 * 参考宿主闭环（reference host loop）：把「自动记忆 + 召回自动注入」落成代码，不靠模型自觉。
 *
 * 每轮 = 取叙事上下文包拼进本轮用户消息头部 → LLM 作答 → 代码自动 ingest 用户原话。
 * 薄层参考实现；引擎核心不动；会话历史由宿主自管（此处为进程内缓冲，重启即新会话，
 * 生产环境的会话存储是宿主自己的职责，不属于记忆引擎）。
 *
 * 缓存友好约定：system 只留稳定人设（永久可缓存）；变动的叙事上下文包 / 危机指令
 * 固定拼在最后一条 user 消息头部——前面的稳定 system + append-only 历史跨轮命中
 * 前缀缓存（如 MiniMax 的 cached_tokens），每轮只有最后一对 user/assistant 尾巴重算。
 * 会话历史只存用户原文，不存带上下文块的包装版（陈旧记忆快照不进历史）。
 *
 * P0-2 修复：危机词表预扫在 LLM 回复生成之前——命中危机信号时先注入安全阀指令，
 * 再生成回复，而非事后 ingest 才发现危机（原顺序：reply → ingest → 危机检测，太晚）。
 */
import { moonshotChat } from '../visualizer/engine/llm'
import type { EngineManager } from './manager'
import type { ContextPacket, IngestResult } from './core'
import { CRISIS_LEXICON } from './core'

/** 稳定段：人设与行为边界，永久可缓存。任何每轮变动的内容都不得插进这段（变动块固定在本轮 user 消息头部）。
 *  导出仅供 bench-cache.ts 复用（保持人设单一来源），行为路径不受影响 */
export const HOST_SYSTEM_PROMPT = `你是一个带长期记忆的陪伴式对话伙伴（衔枝记忆引擎的参考宿主）。
每轮最后一条用户消息开头可能带有〔叙事上下文〕块：那是引擎为你维护的记忆
（进行中的线索、当前理解（带置信度）、近期事件），不是用户原话。
使用纪律：
- 像使用自己的记忆一样自然使用它，不要逐字背诵、不要罗列；
- 上下文里没有的事不要声称记得，被问到时诚实说没有相关记忆；
- 若上下文包含邀请式再提或漂移警示，只在对话容得下反驳的时机自然浮出，不强行插入；
- 基于某条「当前理解」主动提醒/催促/建议前，注意该理解带置信度——它不是事实。`

/** P0-2：危机模式专用指令（正文）——检测到危机信号时替代常规叙事上下文，包进尾部安全阀块 */
const CRISIS_PROMPT = `用户刚刚表达了与自伤/自杀相关的信号。请立即：
- 温暖、在场、不评判、永不推开——检测到风险后冷冰冰拒绝或切断是二次伤害；
- 不说教、不分析原因、不做诊断；
- 如果你知道当地的求助渠道（心理援助热线等），温和地递出来；
- 持续确认用户的安全状态。
这是最高优先级指令，覆盖叙事上下文中的其他指示。`

const MAX_HISTORY = 20

export interface ChatTurnResult {
  reply: string
  /** 本轮实际注入的叙事上下文（透明化：前端可展示「召回了什么」） */
  contextInjected: ContextPacket
  /** 自动 ingest 的引擎侧结果（碎片 id、碰撞判定动作等） */
  ingest: IngestResult
  /** P0-2：本轮是否触发了危机预扫 */
  crisisDetected: boolean
}

/** 进程内会话历史（参考实现，不落盘） */
const histories = new Map<string, { role: 'user' | 'assistant'; content: string }[]>()

export async function chatTurn(manager: EngineManager, userId: string, text: string): Promise<ChatTurnResult> {
  const engine = manager.get(userId)
  // 召回注入：每轮现取现算（引擎侧无陈旧缓存问题），固定拼在本轮 user 消息头部（变动块贴尾）
  const packet = engine.getContextPacket(userId)
  const history = histories.get(userId) ?? []

  // P0-2 修复：危机词表预扫在回复生成之前——命中则注入危机指令，而非事后 ingest 才检测
  // 危机指令同样不进 system：它比叙事包更变动，放尾部块离用户原话更近，安全阀更醒目
  const crisisDetected = CRISIS_LEXICON.test(text)
  const contextBlock = crisisDetected
    ? `〔危机模式 · 安全阀激活〕\n${CRISIS_PROMPT}\n〔安全阀结束〕`
    : `〔叙事上下文 · 引擎注入，非用户原话〕\n${packet.promptText}\n〔上下文结束〕`

  const reply = await moonshotChat([
    { role: 'system', content: HOST_SYSTEM_PROMPT },         // 稳定前缀，跨轮命中缓存
    ...history,                                              // append-only（只存原文），跨轮命中缓存
    { role: 'user', content: `${contextBlock}\n\n${text}` }, // 变动内容固定在末尾
  ], { temperature: crisisDetected ? 0.3 : 0.6, maxTokens: 800 })

  // 自动记忆：用户原话直灌碎片层（零提取失真；高质量提取发生在反刍的认识层，带证据锚定）
  // ingest 内部也会做 CRISIS_LEXICON 检测 → 中止对照窗口，这里不重复
  // withLock 确保 chat 与 reflect/ingest 走同一 per-user 串行锁链，成功时自动持久化
  const ingest = await manager.withLock(userId, (e) => e.ingest(text))
  // 历史只存用户原文（不带上下文块包装）——陈旧记忆快照一旦进历史会越堆越多，也毁缓存
  histories.set(userId, [...history, { role: 'user' as const, content: text }, { role: 'assistant' as const, content: reply }].slice(-MAX_HISTORY))
  return { reply, contextInjected: packet, ingest, crisisDetected }
}
