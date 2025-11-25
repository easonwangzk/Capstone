# Hugging Face Spaces 部署指南

## 📁 文件说明

这个文件夹包含部署到 Hugging Face Spaces 所需的所有文件：

```
huggingface_deployment/
├── app.py                  # 主应用（从 streamlit.py 复制）
├── requirements.txt        # Python 依赖
├── README.md              # Space 主页说明
└── DEPLOYMENT_GUIDE.md    # 本文件（部署指南）
```

---

## 🚀 快速部署步骤

### 步骤 1: 注册 Hugging Face

1. 访问：https://huggingface.co/join
2. 使用 GitHub 或 Google 账号注册（推荐）
3. 验证邮箱

### 步骤 2: 创建新 Space

1. 登录后，点击右上角头像
2. 选择 **"New Space"**
3. 填写信息：
   ```
   Space name: medical-assistant-lora
   License: MIT
   SDK: Streamlit
   Hardware: GPU - A10G Small ($9/月)
   ```
4. 点击 **"Create Space"**

### 步骤 3: 上传文件

#### 方法 A: 网页上传（最简单）

1. 在 Space 页面，点击 **"Files"** 标签
2. 点击 **"Add file"** → **"Upload files"**
3. 上传这 3 个文件：
   - `app.py`
   - `requirements.txt`
   - `README.md`
4. 点击 **"Commit changes to main"**

#### 方法 B: Git 上传（推荐）

```bash
git init
huggingface-cli login

# 1. 克隆你的 Space
git clone https://huggingface.co/spaces/Easonwangzk/medical-assistant-lora
cd medical-assistant-lora

# 2. 复制文件
cp /Users/easonwang/Desktop/Q4/Capstone/huggingface_deployment/* .

# 3. 提交
git add .
git commit -m "Initial deployment"
git push
```

### 步骤 4: 配置 GPU

1. 在 Space 页面，点击 **"Settings"**
2. 找到 **"Hardware"** 部分
3. 选择 **"GPU - A10G Small"**
4. 点击 **"Update"**
5. 绑定信用卡并确认订阅

### 步骤 5: 等待部署

- 查看 **"Logs"** 标签观察进度
- 首次部署需要 10-15 分钟（下载模型）
- 部署成功后会显示 **"Running"** 状态

### 步骤 6: 访问应用

你的应用地址：

```
https://huggingface.co/spaces/你的用户名/medical-assistant-lora
```

---

## 💰 费用说明

### GPU 定价

| 硬件                 | GPU         | VRAM | 价格            |
| -------------------- | ----------- | ---- | --------------- |
| **A10G Small** | NVIDIA A10G | 24GB | **$9/月** |
| CPU Basic            | 无          | -    | 免费            |

### 如何付费

1. Settings → Hardware → GPU - A10G Small
2. 点击 "Upgrade"
3. 绑定信用卡
4. 确认订阅

### 如何取消

- Settings → Hardware → CPU basic (free)
- 随时可以降级或删除 Space

---

## 🔧 故障排查

### 问题 1: 部署失败

**查看日志**：

```
Logs 标签 → 查看错误信息
```

**常见错误**：

1. **模型下载失败**

   ```
   原因：网络问题
   解决：等待几分钟后自动重试
   ```
2. **依赖安装失败**

   ```
   原因：requirements.txt 版本冲突
   解决：检查版本号，确保兼容
   ```
3. **显存不足**

   ```
   原因：GPU 内存不够
   解决：确保选择了 A10G Small (24GB)
   ```

### 问题 2: 应用运行缓慢

**优化建议**：

1. 确保使用 GPU 硬件
2. 检查 `@st.cache_resource` 是否生效
3. 减少 `max_new_tokens` 参数

### 问题 3: 无法访问

**检查步骤**：

1. Space 状态是否为 "Running"
2. Hardware 是否已配置
3. 是否选择了正确的 SDK (Streamlit)

---

## 📊 性能预期

### A10G GPU 性能

| 指标                   | 数值              |
| ---------------------- | ----------------- |
| 模型加载时间           | 30-60 秒          |
| 短回答（100 tokens）   | 1-2 秒            |
| 中等回答（256 tokens） | 2-3 秒            |
| 长回答（512 tokens）   | 3-5 秒            |
| Token 生成速度         | ~60-100 tokens/秒 |

---

## 🔒 隐私设置

### Public vs Private

**Public（公开）**：

- ✅ 免费
- ✅ 任何人都能访问
- ⚠️ 代码公开可见

**Private（私密）**：

- 需要 Pro 订阅
- 只有授权用户能访问
- 代码不公开

### 设置方法

```
Settings → Visibility → 选择 Public 或 Private
```

---

## 📝 更新应用

### 网页更新

1. Files → 点击要修改的文件
2. Edit → 修改内容
3. Commit changes

### Git 更新

```bash
# 修改代码
git add .
git commit -m "Update: 修改说明"
git push
```

自动触发重新部署！

---

## 🎯 测试清单

部署完成后测试：

- [ ] 访问 Space URL 能打开
- [ ] 能看到初始问候消息
- [ ] 输入问题能得到回答
- [ ] 快捷按钮能正常工作
- [ ] 主题切换正常
- [ ] 响应时间在 5 秒内

---

## 💡 额外建议

### 1. 自定义域名（可选）

如果你有自己的域名，可以设置 CNAME：

```
你的域名 → your-space.hf.space
```

### 2. 添加分析

Settings → Analytics → 启用

- 查看访问量
- 用户统计
- 使用趋势

### 3. 分享链接

你的 Space 链接可以分享到：

- 📧 邮件（给老师/同学）
- 📱 社交媒体
- 📄 简历/作品集
- 🎓 课程作业

---

## 🆘 获取帮助

如果遇到问题：

1. **查看官方文档**

   - https://huggingface.co/docs/hub/spaces
2. **社区支持**

   - https://discuss.huggingface.co/
3. **查看示例**

   - https://huggingface.co/spaces
   - 搜索 "streamlit llama" 查看类似项目

---

## ✅ 完成后

你将获得：

- ✅ 永久在线的医疗助手应用
- ✅ 全球可访问的 URL
- ✅ 自动 HTTPS 加密
- ✅ 全球 CDN 加速
- ✅ 免费托管（仅 GPU 付费）
- ✅ 自动备份和版本控制

可以用于：

- 🎓 毕业设计答辩
- 📚 课程作业提交
- 💼 个人作品展示
- 📱 分享给朋友测试

---

## 🎉 开始部署吧！

所有文件已准备好，按照上面的步骤操作即可。

**预计时间**：

- 创建 Space：5 分钟
- 上传文件：2 分钟
- 配置 GPU：2 分钟
- 等待部署：10-15 分钟
- **总计**：约 20-25 分钟

**成本**：$9/月（可随时取消）

祝部署顺利！🚀
