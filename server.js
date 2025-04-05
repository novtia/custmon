import { GoogleGenerativeAI, HarmCategory, HarmBlockThreshold } from "@google/generative-ai";
import fs from "node:fs";
import mime from "mime-types";
import express from "express";
import cors from "cors";
import bodyParser from "body-parser";
import multer from "multer";
import path from "path";
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';

// 加载.env文件中的环境变量
dotenv.config();

// 获取__dirname的ES模块等效值
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
// 项目根目录
const rootDir = path.join(__dirname, '..');

// 配置文件上传存储
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    // 确保上传目录存在
    const uploadDir = path.join(rootDir, 'uploads');
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir, { recursive: true });
    }
    cb(null, uploadDir);
  },
  filename: function (req, file, cb) {
    cb(null, `${Date.now()}-${file.originalname}`);
  }
});

const upload = multer({ storage: storage });

// 工具函数: 将文件转换为Gemini可用的inlineData格式
function fileToGenerativePart(filePath, mimeType) {
  return {
    inlineData: {
      data: Buffer.from(fs.readFileSync(filePath)).toString("base64"),
      mimeType
    }
  };
}

// 多密钥管理
const apiKeyPool = process.env.GEMINI_API_KEYS ? process.env.GEMINI_API_KEYS.split(',').filter(key => key.trim()) : [];
// 如果密钥池为空，则使用单个密钥
if (apiKeyPool.length === 0 && process.env.GEMINI_API_KEY) {
  apiKeyPool.push(process.env.GEMINI_API_KEY);
}

// 当前正在使用的API密钥索引
let currentApiKeyIndex = 0;

// 处理API密钥的实用函数
const apiKeyManager = {
  // 获取当前活跃的API密钥
  getCurrentApiKey: () => {
    return apiKeyPool[currentApiKeyIndex];
  },
  
  // 切换到下一个API密钥
  switchToNextApiKey: () => {
    currentApiKeyIndex = (currentApiKeyIndex + 1) % apiKeyPool.length;
    console.log(`已切换到新的API密钥，当前索引: ${currentApiKeyIndex}`);
    return apiKeyPool[currentApiKeyIndex];
  },
  
  // 获取当前的API客户端
  getGenAIClient: () => {
    return new GoogleGenerativeAI(apiKeyManager.getCurrentApiKey());
  },
  
  // 获取当前生成模型
  getGenerativeModel: (modelConfig = {}) => {
    const genAI = apiKeyManager.getGenAIClient();
    return genAI.getGenerativeModel({
      model: "gemini-2.5-pro-exp-03-25",
      safetySettings: safetySettings,
      ...modelConfig
    });
  },
  
  // 使用自动重试机制执行API调用
  async executeWithRetry(apiCallFn, maxRetries = apiKeyPool.length) {
    let retryCount = 0;
    
    while (retryCount < maxRetries) {
      try {
        return await apiCallFn(apiKeyManager.getGenerativeModel());
      } catch (error) {
        // 检查是否为429错误（请求过多）
        const is429Error = error.message && (
          error.message.includes('429') || 
          error.message.includes('quota') || 
          error.message.includes('rate limit') ||
          error.message.includes('too many requests')
        );
        
        // 如果不是429错误或已经用完所有密钥，则抛出错误
        if (!is429Error || retryCount >= maxRetries - 1) {
          throw error;
        }
        
        console.log(`API密钥受限(429)，尝试切换到下一个密钥: ${retryCount + 1}/${maxRetries}`);
        
        // 切换到下一个API密钥
        apiKeyManager.switchToNextApiKey();
        retryCount++;
      }
    }
    
    throw new Error('所有API密钥都已达到速率限制');
  }
};

// 初始化当前API密钥和生成模型配置
const generationConfig = {
  temperature: 0.1,
  topP: 0.95,
  topK: 64,
  maxOutputTokens: 500,
  responseModalities: [],
  responseMimeType: "text/plain",
};

// 设置安全配置为最低限制
const safetySettings = [
  {
    category: HarmCategory.HARM_CATEGORY_HATE_SPEECH,
    threshold: HarmBlockThreshold.BLOCK_NONE,
  },
  {
    category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
    threshold: HarmBlockThreshold.BLOCK_NONE,
  },
  {
    category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
    threshold: HarmBlockThreshold.BLOCK_NONE,
  },
  {
    category: HarmCategory.HARM_CATEGORY_HARASSMENT,
    threshold: HarmBlockThreshold.BLOCK_NONE,
  },
  {
    category: HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
    threshold: HarmBlockThreshold.BLOCK_NONE,
  },
];

// 创建Express应用
const app = express();
app.use(cors());
app.use(bodyParser.json({ limit: '50mb' }));  // 增加限制以处理较大的JSON请求
app.use('/uploads', express.static(path.join(rootDir, 'uploads')));  // 提供上传文件的静态访问

// 处理图片中的消息内容
function processMessageContent(content) {
  // 如果content是字符串，直接返回文本部分
  if (typeof content === 'string') {
    return [{ text: content }];
  }
  
  // 如果content是数组，处理可能包含的图片
  if (Array.isArray(content)) {
    return content.map(item => {
      if (item.type === 'text') {
        return { text: item.text };
      } else if (item.type === 'image_url') {
        // 处理base64图片格式
        if (item.image_url.url.startsWith('data:')) {
          const matches = item.image_url.url.match(/^data:([^;]+);base64,(.+)$/);
          if (matches && matches.length === 3) {
            const mimeType = matches[1];
            const base64Data = matches[2];
            return {
              inlineData: {
                data: base64Data,
                mimeType
              }
            };
          }
        } 
        // 处理本地文件路径 (如果是通过服务器上传的文件)
        else if (item.image_url.url.startsWith('/uploads/')) {
          const filePath = path.join(rootDir, item.image_url.url);
          const mimeType = mime.lookup(filePath) || 'image/jpeg';
          return fileToGenerativePart(filePath, mimeType);
        }
      }
      return { text: "不支持的内容类型" };
    });
  }
  
  // 默认情况下，将content转换为字符串
  return [{ text: String(content) }];
}

// OpenAI兼容的聊天接口
app.post("/v1/chat/completions", async (req, res) => {
  try {
    const { messages, temperature, max_tokens, stream } = req.body;
    
    // 配置生成参数
    const chatConfig = {
      ...generationConfig,
    };
    
    // 如果提供了temperature，则使用传入的值
    if (temperature !== undefined) {
      chatConfig.temperature = temperature;
    }
    
    // 如果提供了max_tokens，则使用传入的值
    if (max_tokens !== undefined) {
      chatConfig.maxOutputTokens = max_tokens;
    }
    
    // 转换OpenAI消息格式为Gemini格式
    const history = [];
    for (const message of messages.slice(0, -1)) {  // 排除最后一条消息，作为当前输入处理
      const { role, content } = message;
      // 转换role: user -> user, assistant -> model, system -> user (with prefix)
      const geminiRole = role === "assistant" ? "model" : "user";
      
      // 处理系统消息
      let processedContent;
      if (role === "system") {
        processedContent = [{ text: `[SYSTEM]: ${content}` }];
      } else {
        processedContent = typeof content === 'string' ? [{ text: content }] : processMessageContent(content);
      }
      
      history.push({
        role: geminiRole,
        parts: processedContent,
      });
    }
    
    // 获取最后一条消息
    const lastMessage = messages[messages.length - 1];
    const lastMessageParts = processMessageContent(lastMessage.content);
    
    // 检查是否请求了流式输出
    if (stream) {
      // 设置适当的响应头
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      
      // 生成唯一ID
      const responseId = `chatcmpl-${Date.now()}`;
      const created = Math.floor(Date.now() / 1000);
      
      try {
        // 发送开始事件
        const startEvent = {
          id: responseId,
          object: "chat.completion.chunk",
          created,
          model: "gemini-2.5-pro-exp-03-25",
          choices: [
            {
              index: 0,
              delta: { role: "assistant" },
              finish_reason: null
            }
          ]
        };
        res.write(`data: ${JSON.stringify(startEvent)}\n\n`);
        
        // 使用带有自动重试的API调用
        await apiKeyManager.executeWithRetry(async (model) => {
          const chatSession = model.startChat({
            generationConfig: chatConfig,
            history: history,
          });
          
          const streamResult = await chatSession.sendMessageStream(lastMessageParts);
          
          // 初始化内容
          let contentSoFar = '';
          
          // 处理每个流式响应块
          for await (const chunk of streamResult.stream) {
            const chunkText = chunk.text();
            
            if (chunkText) {
              contentSoFar += chunkText;
              
              const chunkEvent = {
                id: responseId,
                object: "chat.completion.chunk",
                created,
                model: "gemini-2.5-pro-exp-03-25",
                choices: [
                  {
                    index: 0,
                    delta: { content: chunkText },
                    finish_reason: null
                  }
                ]
              };
              
              res.write(`data: ${JSON.stringify(chunkEvent)}\n\n`);
            }
          }
          
          // 发送完成事件
          const finishEvent = {
            id: responseId,
            object: "chat.completion.chunk",
            created,
            model: "gemini-2.5-pro-exp-03-25",
            choices: [
              {
                index: 0,
                delta: {},
                finish_reason: "stop"
              }
            ]
          };
          res.write(`data: ${JSON.stringify(finishEvent)}\n\n`);
          res.write(`data: [DONE]\n\n`);
          res.end();
        });
      } catch (error) {
        console.error("Stream error:", error);
        
        // 发送错误事件
        const errorEvent = {
          error: {
            message: error.message,
            type: "server_error"
          }
        };
        res.write(`data: ${JSON.stringify(errorEvent)}\n\n`);
        res.write(`data: [DONE]\n\n`);
        res.end();
      }
    } else {
      // 非流式响应，使用带有自动重试的API调用
      try {
        const result = await apiKeyManager.executeWithRetry(async (model) => {
          const chatSession = model.startChat({
            generationConfig: chatConfig,
            history: history,
          });
          
          return await chatSession.sendMessage(lastMessageParts);
        });
        
        // 格式化为OpenAI响应格式
        const response = {
          id: `chatcmpl-${Date.now()}`,
          object: "chat.completion",
          created: Math.floor(Date.now() / 1000),
          model: "gemini-2.5-pro-exp-03-25",
          choices: [
            {
              index: 0,
              message: {
                role: "assistant",
                content: result.response.text(),
              },
              finish_reason: "stop",
            },
          ],
          usage: {
            prompt_tokens: -1, // Gemini不直接提供token计数
            completion_tokens: -1,
            total_tokens: -1,
          },
        };
        
        res.json(response);
      } catch (error) {
        console.error("Error:", error);
        res.status(500).json({
          error: {
            message: error.message,
            type: "internal_server_error",
          }
        });
      }
    }
  } catch (error) {
    console.error("Error:", error);
    res.status(500).json({
      error: {
        message: error.message,
        type: "internal_server_error",
      }
    });
  }
});

// 文件上传端点 - 支持多文件上传
app.post('/upload-files', upload.array('files', 10), (req, res) => {
  try {
    const uploadedFiles = req.files.map(file => ({
      filename: file.filename,
      originalname: file.originalname,
      mimetype: file.mimetype,
      path: file.path,
      url: `/uploads/${file.filename}`
    }));
    
    res.json({
      success: true,
      files: uploadedFiles
    });
  } catch (error) {
    console.error('文件上传错误:', error);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

// 直接进行图像分析的API端点
app.post('/backends/chat-completions/generate', upload.array('images', 10), async (req, res) => {
  try {
    const { prompt } = req.body;
    const uploadedFiles = req.files || [];
    
    if (!prompt) {
      return res.status(400).json({
        success: false,
        error: "必须提供prompt参数"
      });
    }
    
    // 准备请求内容
    const messageParts = [{ text: prompt }];
    
    // 处理上传的图片
    for (const file of uploadedFiles) {
      const mimeType = file.mimetype || mime.lookup(file.path) || 'image/jpeg';
      const imagePart = fileToGenerativePart(file.path, mimeType);
      messageParts.push(imagePart);
    }
    
    // 处理可能的base64编码图片
    const images = req.body.images || [];
    for (const image of images) {
      if (image.data && image.mimeType) {
        messageParts.push({
          inlineData: {
            data: image.data,
            mimeType: image.mimeType
          }
        });
      }
    }
    
    // 调用Gemini API
    const result = await apiKeyManager.executeWithRetry(async (model) => {
      return await model.generateContent(messageParts);
    });
    
    res.json({
      success: true,
      response: result.response.text()
    });
  } catch (error) {
    console.error('图像分析错误:', error);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

// 健康检查端点
app.get("/health", (req, res) => {
  res.json({ 
    status: "ok",
    apiKeys: {
      total: apiKeyPool.length,
      current: currentApiKeyIndex
    }
  });
});

// 静态文件服务
// 提供前端静态资源
const distPath = path.join(rootDir, 'dist');
app.use(express.static(distPath));

// 获取端口配置，默认为3000
const PORT = process.env.PORT || 3000;

// 启动服务器
app.listen(PORT, () => {
  console.log(`服务器已启动，正在监听端口 ${PORT}`);
  console.log(`- API服务地址: http://localhost:${PORT}/v1/chat/completions`);
  console.log(`- 健康检查接口: http://localhost:${PORT}/health`);
  console.log(`- 静态文件服务: http://localhost:${PORT}/`);
  console.log(`- 文件上传地址: http://localhost:${PORT}/upload-files`);
  console.log(`使用API密钥数量: ${apiKeyPool.length}`);
});
