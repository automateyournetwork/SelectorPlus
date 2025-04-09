# Email MCP Server

MCP Server for the Email.

## Setup

### NPX

```json
{
    "mcpServers": {
        "email-mcp": {
            "command": "npx",
            "args": [
                "-y",
                "email-mcp"
            ],
            "env": {
                "EMAIL_HOST": "<your SMTP server>",
                "EMAIL_PORT": "<your SMTP port>",
                "EMAIL_SSL": "<true or false>",
                "EMAIL_ACCOUNT": "<your email account>",
                "EMAIL_PASSWORD": "<password or app password>"
            }
        }
    }
}
```