FROM node:lts-alpine

# Create app directory
WORKDIR /app

# Install app dependencies
COPY package*.json ./
RUN npm install --ignore-scripts

# Copy source code
COPY . .

# Build the TypeScript project
RUN npm run build

# Run the compiled JS entry point
CMD ["node", "build/index.js"]
