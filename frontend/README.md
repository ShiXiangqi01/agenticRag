# Agentic RAG - Frontend

一个 React + TypeScript 的前端应用，用于与 Agentic RAG 系统交互。

## 功能特性

- 💬 **实时聊天** - 与 AI 代理进行实时对话
- 🖼️ **图像支持** - 上传图像进行分析
- 📸 **图像结果显示** - 查看相关的检索图像
- 🔄 **会话管理** - 管理多个对话会话
- 🤖 **模型选择** - 支持多个 AI 模型
- 🎨 **现代 UI** - 使用 Tailwind CSS 和自定义组件

## 技术栈

- **框架**: React 18 + TypeScript
- **样式**: Tailwind CSS
- **状态管理**: Zustand
- **HTTP 客户端**: Axios
- **构建工具**: Vite
- **图标**: Lucide React

## 快速开始

### 安装

```bash
cd frontend
npm install
```

### 开发

```bash
npm run dev
```

应用将在 `http://localhost:5173` 运行，API 代理到 `http://localhost:8000`

### 构建

```bash
npm run build
```

### 类型检查

```bash
npm run type-check
```

## 项目结构

```
frontend/
├── src/
│   ├── components/          # React 组件
│   │   ├── ChatHistory.tsx  # 聊天历史显示
│   │   ├── MessageInput.tsx # 消息输入框
│   │   ├── MessageItem.tsx  # 单条消息
│   │   ├── ModelSelector.tsx# 模型选择器
│   │   ├── Sidebar.tsx      # 侧边栏
│   │   └── ...
│   ├── hooks/              # 自定义 Hooks
│   │   ├── useSendMessage.ts
│   │   ├── useImageUpload.ts
│   │   └── useInitialize.ts
│   ├── pages/              # 页面组件
│   │   └── ChatPage.tsx
│   ├── services/           # API 服务
│   │   └── api.ts
│   ├── store/              # Zustand 状态管理
│   │   └── chat.ts
│   ├── types/              # TypeScript 类型定义
│   │   └── api.ts
│   ├── App.tsx             # 主组件
│   ├── main.tsx            # 入口文件
│   └── index.css           # 全局样式
├── index.html              # HTML 模板
├── package.json            # 依赖配置
├── tsconfig.json           # TypeScript 配置
├── vite.config.ts          # Vite 配置
└── tailwind.config.ts      # Tailwind CSS 配置
```

## API 集成

前端与以下后端 API 交互：

- `POST /agents/send_message` - 发送消息到代理
- `GET /agents/sessions` - 获取所有会话
- `GET /agents/available_models` - 获取可用模型
- `POST /user_image/upload` - 上传用户图像
- `GET /user_image/{imageId}` - 获取用户图像

## 状态管理

使用 Zustand 管理应用状态：

- `messages` - 聊天消息列表
- `currentSessionId` - 当前会话 ID
- `selectedModel` - 选中的 AI 模型
- `isLoading` - 加载状态
- `error` - 错误消息
- `sessions` - 所有会话
- `availableModels` - 可用的模型列表

## 环境变量

可选的环境变量：

- `VITE_API_BASE_URL` - API 基础 URL（默认: `/api`）

## 浏览器支持

- Chrome/Edge 最新版本
- Firefox 最新版本
- Safari 最新版本

## 许可证

MIT
