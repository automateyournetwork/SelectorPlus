# Use Node with TypeScript pre-installed
FROM node:20

# Create working directory
WORKDIR /app

# Copy dependency files
COPY package*.json tsconfig.json ./

# Install dependencies
RUN npm install

# Copy the rest of the source code
COPY . .

# Compile TypeScript → JavaScript
RUN npm run build

# Install dotenv-cli to load .env file at runtime
RUN npm install -g dotenv-cli

# Run the built JavaScript file in /build
ENTRYPOINT ["dotenv", "--"]
CMD ["node", "build/index.js"]
