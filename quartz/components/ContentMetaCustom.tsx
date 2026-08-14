import { QuartzComponentConstructor } from "./types"
import { i18n } from "../i18n"

/** 简易阅读时间估算（CJK 400 字/分钟 + 英文 200 词/分钟），避免引入 reading-time 依赖 */
function estimateMinutes(text: string): number {
  const cjk = (text.match(/[\u4e00-\u9fff\u3400-\u4dbf]/g) || []).length
  const words = text.split(/\s+/).filter(Boolean).length
  return Math.max(1, Math.ceil(cjk / 400 + words / 200))
}

function formatDate(d: Date, locale: string): string {
  return d.toLocaleDateString(locale, { year: "numeric", month: "short", day: "2-digit" })
}

const css = `
.content-meta {
  margin-top: 0;
  color: var(--darkgray);
}
.content-meta .meta-date {
  margin-right: 4px;
}
.content-meta .meta-label {
  color: var(--gray);
  margin-right: 4px;
}
.content-meta[show-comma="true"] > *:not(:last-child) {
  margin-right: 8px;
}
.content-meta[show-comma="true"] > *:not(:last-child)::after {
  content: ",";
}
`

/**
 * 自定义 ContentMeta：同时显示发布时间（published）与更新时间（modified/updated）。
 * 覆盖 @quartz-community/content-meta 默认组件（默认只显示 modified）。
 * 日期数据由 @quartz-community/created-modified-date 插件从 frontmatter 解析提供。
 */
const ContentMeta: QuartzComponentConstructor = () => {
  function ContentMetaComponent({ cfg, fileData, displayClass }: any) {
    const text = fileData.text
    if (!text) return null
    const locale = cfg.locale || "en-US"
    const dates = fileData.dates as Record<string, Date> | undefined
    const segments = []

    if (dates?.published) {
      segments.push(
        <span class="meta-label">发布</span>,
        <time class="meta-date" datetime={dates.published.toISOString()}>
          {formatDate(dates.published, locale)}
        </time>,
      )
    }
    if (dates?.modified) {
      segments.push(
        <span class="meta-label">更新</span>,
        <time class="meta-date" datetime={dates.modified.toISOString()}>
          {formatDate(dates.modified, locale)}
        </time>,
      )
    }
    const minutes = estimateMinutes(text)
    const i18nData = i18n(locale)
    segments.push(
      <span>{i18nData.components.contentMeta.readingTime({ minutes })}</span>,
    )

    return (
      <p show-comma={true} class={[displayClass, "content-meta"].filter(Boolean).join(" ")}>
        {segments}
      </p>
    )
  }
  ContentMetaComponent.css = css
  return ContentMetaComponent
}

export default ContentMeta
