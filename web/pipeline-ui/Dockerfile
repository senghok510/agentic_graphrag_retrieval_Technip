# Multi-stage build producing a minimal runtime image via Next.js "standalone"
# output (next.config.mjs sets output: "standalone"). Build context is this
# directory (web/pipeline-ui/).
#
#   docker build -t tender-pipeline-ui:latest -f web/pipeline-ui/Dockerfile web/pipeline-ui/

FROM node:20-alpine AS deps
# Patch OS-level packages baked into the base image tag (Alpine ships CVE
# fixes continuously; the node:20-alpine tag itself is only rebuilt
# periodically, so its layer can lag behind).
RUN apk update && apk upgrade --no-cache
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

FROM node:20-alpine AS builder
RUN apk update && apk upgrade --no-cache
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN npm run build

FROM node:20-alpine AS runner
RUN apk update && apk upgrade --no-cache
WORKDIR /app
ENV NODE_ENV=production
ENV PORT=3000
# No public/ directory exists yet in this app -- add
# `COPY --from=builder /app/public ./public` back if one is created later.
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static

EXPOSE 3000
CMD ["node", "server.js"]
