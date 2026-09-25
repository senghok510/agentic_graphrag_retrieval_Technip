/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Standalone output traces the minimal set of files/deps needed to run and
  // copies them into .next/standalone -- lets the Docker runtime stage skip
  // node_modules entirely (see web/pipeline-ui/Dockerfile).
  output: "standalone",
  // Proxy /api/* to the FastAPI pipeline backend so the browser talks to one origin
  // (avoids CORS entirely and keeps SSE streaming through Next's dev server).
  async rewrites() {
    const api = process.env.PIPELINE_API_URL || "http://localhost:7070";
    return [{ source: "/api/:path*", destination: `${api}/:path*` }];
  },
};

export default nextConfig;
