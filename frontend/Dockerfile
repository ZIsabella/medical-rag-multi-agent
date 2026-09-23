FROM node:20-slim AS deps
WORKDIR /app

ENV NEXT_TELEMETRY_DISABLED=1

# فقط فایل‌های وابستگی
COPY package.json package-lock.json* ./

# نصب کامل (شامل devDependencies)
# اگر package-lock ندارید، npm install کار می‌کند و devDeps را نصب می‌کند
RUN npm install --include=dev --legacy-peer-deps


FROM node:20-slim AS builder
WORKDIR /app

ENV NEXT_TELEMETRY_DISABLED=1

COPY --from=deps /app/node_modules ./node_modules
COPY . .

# بیلد
RUN npm run build


FROM node:20-slim AS runner
WORKDIR /app

ENV NODE_ENV=production
ENV PORT=3000
ENV HOSTNAME=0.0.0.0
ENV NEXT_TELEMETRY_DISABLED=1

COPY --from=builder /app/package.json ./package.json
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/public ./public
COPY --from=builder /app/.next ./.next

EXPOSE 3000

CMD ["node", "node_modules/next/dist/bin/next", "start"]
