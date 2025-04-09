#!/usr/bin/env node
"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
var _a, _b;
Object.defineProperty(exports, "__esModule", { value: true });
const mcp_js_1 = require("@modelcontextprotocol/sdk/server/mcp.js");
const nodemailer_1 = __importDefault(require("nodemailer"));
const zod_1 = require("zod");
const stdio_js_1 = require("@modelcontextprotocol/sdk/server/stdio.js");
// 创建服务器实例
const email_server = new mcp_js_1.McpServer({
    name: "email",
    version: "1.0.0",
    capabilities: {
        resources: {},
        tools: {},
    },
});
// 创建邮件传输器
const transporter = nodemailer_1.default.createTransport({
    host: (_a = process.env.EMAIL_HOST) !== null && _a !== void 0 ? _a : "smtp.gmail.com", // 替换为你的 SMTP 服务器
    port: (_b = process.env.EMAIL_PORT) !== null && _b !== void 0 ? _b : 587,
    secure: process.env.EMAIL_SSL === "true" ? true : false,
    auth: {
        user: process.env.EMAIL_ACCOUNT, // 替换为你的邮箱
        pass: process.env.EMAIL_PASSWORD, // 替换为你的密码或应用专用密码
    },
});
// 发送邮件的辅助函数
async function sendEmail(options) {
    try {
        await transporter.sendMail({
            from: process.env.EMAIL_ACCOUNT, // 替换为你的邮箱
            to: options.to,
            subject: options.subject,
            text: options.text,
        });
        return {
            status: true,
            message: "邮件发送成功",
        };
    }
    catch (error) {
        const errorMessage = error.message;
        console.error("Error sending email:", errorMessage);
        return {
            status: false,
            message: errorMessage,
        };
    }
}
// 注册发送邮件工具
email_server.tool("send-email", "发送邮件", {
    to: zod_1.z.string().describe("邮件收件人"),
    subject: zod_1.z.string().describe("邮件主题"),
    text: zod_1.z.string().describe("邮件内容"),
}, async ({ to, subject, text }) => {
    if (!to || !subject || !text) {
        return {
            content: [
                {
                    type: "text",
                    text: "邮件参数不完整，请提供收件人、主题和内容",
                },
            ],
        };
    }
    const response = await sendEmail({ to, subject, text });
    return {
        content: [
            {
                type: "text",
                text: response.message,
            },
        ],
    };
});
async function runServer() {
    const transport = new stdio_js_1.StdioServerTransport();
    await email_server.connect(transport);
    console.error("Email MCP Server running on stdio");
}
runServer().catch((error) => {
    console.error("Fatal error running server:", error);
    process.exit(1);
});
