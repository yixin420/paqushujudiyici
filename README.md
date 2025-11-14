# Mastodon 中文公开帖爬取脚本

该仓库包含一个 Python 脚本 `mastodon_chinese_scraper.py`，用于从 Mastodon 实例（默认 `https://m.cmx.im`）批量获取指定时间范围内的中文公开帖子（含简体、繁体），并按周保存最新的前 100 条帖子。

## 运行环境

- Python 3.9+
- 依赖库：
  - `requests`
  - `beautifulsoup4`
  - `schedule`

安装依赖：

```bash
python -m pip install -r requirements.txt
```

如未使用 `requirements.txt`，也可直接安装：

```bash
python -m pip install requests beautifulsoup4 schedule
```

## 使用方法

1. **调整配置**  
   打开 `mastodon_chinese_scraper.py` 顶部的配置段落，根据需要修改：
   - `BASE_URL`：目标 Mastodon 实例，例如 `https://mastodon.social`
   - `START_TIME_UTC`、`END_TIME_UTC`：UTC 时间范围（务必保留 `timezone.utc`）
   - `WEEKLY_LIMIT`：每周保留的帖子数量（默认 100）
   - `OUTPUT_DIR`：输出目录
   - `AUTH_BEARER_TOKEN`：若匿名请求返回 401，可填入访问令牌
   - `USE_DAILY_ROLLING_WINDOW`：默认为 `True`，表示每次运行只抓取“当天”（UTC 00:00:00 至 23:59:59）的公开帖，避免重复抓取历史数据；若想指定固定时间段，可将其置为 `False` 并修改 `START_TIME_UTC`/`END_TIME_UTC`

2. **运行脚本并启用定时任务**  

```bash
python mastodon_chinese_scraper.py
```

脚本会立即执行一次爬取，然后保持常驻，通过 `schedule` 库每分钟检查当前 UTC 时间，在每日 `23:55 UTC` 自动再次触发 `MastodonChineseScraper`（包含 `run()` 和 `print_sample_posts()` 流程）。默认开启的 `USE_DAILY_ROLLING_WINDOW` 会让每次定时运行只覆盖当天（UTC）的时间窗口，从而避免重复抓取历史记录。使用 `Ctrl+C` 可手动停止；若需长期运行，建议放入 `tmux`、`screen` 或通过 `systemd`/容器后台运行。

> 若需要改为本地时间，可调整服务器时区或将 `SCHEDULED_UTC_HOUR` / `SCHEDULED_UTC_MINUTE` 改成对应的 UTC 值；`schedule` 调度逻辑会继续以 UTC 判断触发时间。

脚本仍会自动遵守接口每秒至多一次请求、429 时读取 `X-RateLimit-Reset` 并等待后重试，同时对网络异常执行最多 3 次重试（间隔 2 秒）。

3. **输出结果**  
   - 每个周的数据保存为 `YYYY_weekNN_chinese_posts.json`，位于 `OUTPUT_DIR`（默认 `data/`）
   - JSON 中仅包含以下字段：`id`、`account_id`、`username`、`display_name`、`note`、`created_at`、`url`、`content`
   - 运行结束会打印前 5 条中文帖子的调试信息，便于快速验证

## 常见问题

- **401 Unauthorized**：实例要求认证，设置 `AUTH_BEARER_TOKEN`
- **429 Too Many Requests**：脚本会自动等待重置时间，必要时可增大 `MIN_REQUEST_INTERVAL_SECONDS`
- **503 Service Unavailable**：实例暂时不可用，脚本会重试，必要时稍后重试
- **没有中文数据**：尝试更换实例、放宽时间范围或降低每周数量限制

## 文件结构

- `mastodon_chinese_scraper.py`：核心爬虫脚本
- `data/`：运行后生成的周度 JSON 输出目录（脚本会自动创建）
