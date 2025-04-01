# ----------------------
# Stage 1: node_builder
# ----------------------
    FROM node:22-bookworm-slim AS node_builder

    WORKDIR /app
    
    ENV PLAYWRIGHT_BROWSERS_PATH=0
    
    COPY package*.json ./
    COPY package-lock.json ./
    RUN --mount=type=cache,target=/root/.npm npm ci
    
    COPY tsconfig.json ./
    COPY src ./src
    
    RUN npm run build
    
    # ----------------------
    # Stage 2: app_release
    # ----------------------
    FROM debian:bookworm-slim AS app_release
    
    ENV PATH="$PATH:/app/node_modules/.bin"
    ENV PLAYWRIGHT_BROWSERS_PATH=0
    
    RUN apt-get update && apt-get install -y \
        ca-certificates \
        libx11-6 \
        libxcomposite1 \
        libxcursor1 \
        libxdamage1 \
        libxext6 \
        libxfixes3 \
        libxi6 \
        libxrandr2 \
        libxrender1 \
        libxtst6 \
        fonts-freefont-ttf \
        libnss3 \
        libasound2 \
        libatk-bridge2.0-0 \
        libgtk-3-0 \
        chromium \
        udev \
        libpng16-16 \
        curl
    
    #install node and npm.
    RUN curl -sL https://deb.nodesource.com/setup_22.x | bash - && \
        apt-get install -y nodejs
    
    # Verify and set the PATH explicitly
    RUN echo "PATH: $PATH"
    RUN which node
    RUN which npm
    
    WORKDIR /app
    
    COPY --from=node_builder /app/build ./build
    COPY --from=node_builder /app/package*.json ./
    COPY --from=node_builder /app/node_modules ./node_modules
    
    RUN npx playwright install chromium
    
    ENV NODE_ENV=production
    
    #Use the full path to node.
    CMD ["/usr/bin/node", "build/index.js"]