import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// 프론트는 wheel에 정적으로 실려 hub가 서빙한다(기획서 §9).
// 초기 로드 예산 3 MB gzip — 무거운 라이브러리는 지연 로드 청크로 뺀다.
export default defineConfig({
  plugins: [react()],
  build: { outDir: "../src/torchflow/hub/static", emptyOutDir: true },
  server: { proxy: { "/api": "http://127.0.0.1:8765", "/ws": { target: "ws://127.0.0.1:8765", ws: true } } },
});
