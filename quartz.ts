import { loadQuartzConfig, loadQuartzLayout } from "./quartz/plugins/loader/config-loader"
import { componentRegistry } from "./quartz/components/registry"
import ContentMetaCustom from "./quartz/components/ContentMetaCustom"

// 覆盖 @quartz-community/content-meta 默认组件：同时显示发布时间 + 更新时间。
// 必须在 loadQuartzConfig 之前注册；componentLoader 检测到同名已注册会跳过插件覆盖。
// key 必须与插件实际注册名一致：yaml source 为 "@quartz-community/content-meta"，
// componentLoader 会注册 pluginName / exportName / 纯名 三种 key。
// 注：pluginName/exportName（"@quartz-community/content-meta/ContentMeta"）会被插件无条件覆盖，无需注册。
const manifest = {
  name: "content-meta",
  displayName: "Content Meta",
  description: "Show published and updated dates",
  version: "1.0.0",
  defaultPosition: "beforeBody",
  defaultPriority: 20,
}
for (const key of [
  "@quartz-community/content-meta", // pluginName（buildLayoutForEntries 主查找 key）
  "ContentMeta", // exportName 纯名
  "content-meta", // kebab 名
]) {
  if (!componentRegistry.get(key)) {
    componentRegistry.register(key, ContentMetaCustom, "local-custom", manifest)
  }
}

const config = await loadQuartzConfig()
export default config
export const layout = await loadQuartzLayout()
