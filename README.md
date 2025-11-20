# Mastodon 中文公开帖爬取脚本

该仓库提供脚本 `mastodon_chinese_scraper.py`，用于从指定 Mastodon 实例（默认 `https://m.cmx.im`）抓取 **当前 UTC 日** 内的全部中文（简/繁）公开帖子。脚本支持实时监听直到当天结束，并且会按照 `爬取日期_实例域名_(当天抓取总数).json` 的格式落盘。

## 运行环境

- Python 3.9+
- 依赖库
  - `requests`
  - `beautifulsoup4`

安装依赖：

```bash
python -m pip install -r requirements.txt
```

## 使用方法

1. **调整配置**  
   在脚本顶部可修改：
   - `BASE_URL`：目标实例，默认 `https://m.cmx.im`
   - `START_TIME_UTC` / `END_TIME_UTC`：抓取窗口（默认自动覆盖当前 UTC 日 00:00:00~23:59:59）
   - `OUTPUT_DIR`：存储目录，默认 `data/`
   - `AUTH_BEARER_TOKEN`：若实例关闭匿名访问，可在此填入 `read:statuses` 权限的 token
   - `WAIT_FOR_DAY_COMPLETION`：是否等待直到 `END_TIME_UTC`，以确保拿到全天数据（默认 `True`）
   - `LIVE_POLL_INTERVAL_SECONDS`：实时轮询间隔（默认 60 秒，可视流量和速率限制调整）

2. **运行脚本**

```bash
python mastodon_chinese_scraper.py
```

脚本会自动处理 401/429/503 等常见响应、遵循 `limit=40` 的分页，并利用 `max_id`（向过去翻页）和 `min_id`（实时向前追踪）确保覆盖完整一天的时间线。

3. **输出结果**
   - 每个日期会生成一个 JSON 文件，例如 `20250115_m-cmx-im_(4321).json`
   - JSON 字段包含 `id`、`account_id`、`username`、`display_name`、`note`、`created_at`、`url`、`content`
   - 汇总文件 `daily_summary.csv` 会同步更新当天总量
   - 运行结束打印至多 5 条示例数据，方便快速验证

## 时间线 API 速览

脚本默认使用 **Public timeline** (`GET /api/v1/timelines/public`)。你也可以基于 Mastodon 时间线 API 文档调整为其他端点：

- **Public timeline**：支持 `local`、`remote`、`only_media`、`max_id`、`min_id`、`since_id`、`limit`
- **Hashtag timeline** (`/timelines/tag/:hashtag`)：额外支持 `any[]`、`all[]`、`none[]`
- **Home timeline** (`/timelines/home`)：需要用户 token + `read:statuses`
- **List timeline** (`/timelines/list/:list_id`)：需要用户 token + `read:lists`
- **Link timeline** (`/timelines/link?url=...`)：过滤当前热门链接

所有时间线均支持 `min_id` 与 `max_id` 同时使用（≥3.3.0），脚本即基于该机制来“回溯 + 实时追踪”当天的完整帖子集。

## 常见问题

- **401 Unauthorized**：实例禁用匿名访问时请配置 `AUTH_BEARER_TOKEN`
- **429 Too Many Requests**：脚本会自动读取 `X-RateLimit-Reset` 并等待，必要时可提高 `REQUEST_DELAY_SECONDS`
- **脚本运行时间较长**：`WAIT_FOR_DAY_COMPLETION=True` 时会一直轮询到当天结束，若只需即时快照可改为 `False`
- **没有中文数据**：确认实例确实存在中文内容，必要时调整 `BASE_URL` 或放宽抓取窗口

## 文件结构

- `mastodon_chinese_scraper.py`：核心爬虫脚本
- `requirements.txt`：依赖声明
- `data/`：按天输出的 JSON 及 `daily_summary.csv`（运行时自动创建）
