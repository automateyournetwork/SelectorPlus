FROM node:22.12-alpine AS builder

COPY . /app

WORKDIR /app

RUN --mount=type=cache,target=/root/.npm npm install && \
    npx tsc --noEmit --skipLibCheck && \
    npm run build

FROM node:22-alpine AS release

WORKDIR /app

COPY --from=builder /app/dist /app/dist
COPY --from=builder /app/node_modules /app/node_modules

ENTRYPOINT ["node", "dist/index.js"]