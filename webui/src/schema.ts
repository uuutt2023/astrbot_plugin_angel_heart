/** _conf_schema.json 单项元数据（前端渲染用最小集） */

/** 下拉可选项：字符串或 {label, value} */
export type SchemaOption = string | { label: string; value: string }

export interface SchemaMeta {
  type?: string
  description?: string
  hint?: string
  items?: Record<string, SchemaMeta>
  _special?: string
  default?: unknown
  editor_language?: string
  /** 枚举可选项（保存值） */
  options?: SchemaOption[]
  /** 下拉展示文本，与 options 按索引对应 */
  labels?: string[]
  /** 选项下方说明，与 options 按索引对应 */
  subtitles?: string[]
}
