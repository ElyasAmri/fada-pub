import fs from "node:fs";
import path from "node:path";
import type { NextConfig } from "next";

function loadParentDotEnv() {
  const envPath = path.resolve(process.cwd(), "..", ".env.local");
  if (!fs.existsSync(envPath)) {
    return;
  }

  const lines = fs.readFileSync(envPath, "utf8").split(/\r?\n/);
  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) {
      continue;
    }

    const equalsIndex = trimmed.indexOf("=");
    if (equalsIndex === -1) {
      continue;
    }

    const key = trimmed.slice(0, equalsIndex).trim();
    if (!key || process.env[key]) {
      continue;
    }

    const rawValue = trimmed.slice(equalsIndex + 1).trim();
    const value =
      (rawValue.startsWith('"') && rawValue.endsWith('"')) ||
      (rawValue.startsWith("'") && rawValue.endsWith("'"))
        ? rawValue.slice(1, -1)
        : rawValue;

    process.env[key] = value;
  }
}

loadParentDotEnv();

const nextConfig: NextConfig = {
  /* config options here */
  reactCompiler: true,
};

export default nextConfig;
