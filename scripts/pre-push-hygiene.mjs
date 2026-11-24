#!/usr/bin/env node
/**
 * Pre-Push Security, Secret, Bloat & Hygiene Gate
 * Usage:
 *   node scripts/pre-push-hygiene.mjs <repoDir> [--strict-docs]
 */
import fs from "node:fs";
import path from "node:path";
import { spawnSync } from "node:child_process";

const args = process.argv.slice(2);
if (args.length < 1) {
  console.error("Usage: node pre-push-hygiene.mjs <repoDir> [--strict-docs]");
  process.exit(2);
}

const repoDir = path.resolve(args[0]);
const strictDocs = args.includes("--strict-docs");
const MAX_FILE_BYTES = 10 * 1024 * 1024;

const FORBIDDEN_EXTS = new Set([
  ".safetensors", ".pt", ".pth", ".bin", ".onnx", ".gguf", ".ckpt",
  ".sqlite", ".sqlite3", ".db", ".db-wal", ".db-shm",
  ".pem", ".key", ".p12", ".pfx", ".env",
]);

const SKIP_DIRS = new Set([
  ".git", "node_modules", ".venv", ".venv-dl", "venv",
  "__pycache__", ".pytest_cache", ".ruff_cache", "dist", ".next", "target",
]);

const SECRET_PATTERNS = [
  { id: "GITHUB_CLASSIC_PAT", regex: /\bghp_[A-Za-z0-9]{30,}\b/g },
  { id: "GITHUB_FINE_PAT", regex: /\bgithub_pat_[A-Za-z0-9_]{30,}\b/g },
  { id: "GITHUB_OAUTH", regex: /\bgho_[A-Za-z0-9]{30,}\b/g },
  { id: "OPENAI_KEY", regex: /\bsk-(?:proj-)?[A-Za-z0-9_-]{32,}\b/g },
  { id: "GOOGLE_API_KEY", regex: /\bAIza[0-9A-Za-z_-]{35}\b/g },
  { id: "AWS_ACCESS_KEY", regex: /\b(?:AKIA|ASIA)[0-9A-Z]{16}\b/g },
  { id: "SLACK_TOKEN", regex: /\bxox[baprs]-[0-9A-Za-z-]{10,}\b/g },
  { id: "PRIVATE_KEY_BLOCK", regex: /-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----/g },
];

const LOCAL_PATH_REGEX = /[A-Za-z]:\\{1,2}Users\\{1,2}ardhy(?:\\{1,2}[^\s"'`<>)]*)?|C:\/Users\/ardhy(?:\/[^\s"'`<>)]*)?/gi;

const TEXT_EXTS = new Set([
  ".md", ".txt", ".py", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx",
  ".json", ".yml", ".yaml", ".toml", ".ini", ".cfg", ".sh", ".ps1",
  ".html", ".css", ".xml", ".svg", ".dockerfile", ".gitignore", ".env",
]);

function isIgnoredByGit(relPosix) {
  if (!fs.existsSync(path.join(repoDir, ".git"))) return false;
  const res = spawnSync("git", ["-C", repoDir, "check-ignore", "-q", relPosix], { stdio: "ignore" });
  return res.status === 0;
}

function listFiles(dir) {
  const results = [];
  function walk(cur) {
    for (const ent of fs.readdirSync(cur, { withFileTypes: true })) {
      if (SKIP_DIRS.has(ent.name)) continue;
      const full = path.join(cur, ent.name);
      const rel = path.relative(repoDir, full).replace(/\\/g, "/");
      if (isIgnoredByGit(rel)) continue;
      if (ent.isDirectory()) {
        walk(full);
      } else if (ent.isFile()) {
        results.push({ full, rel });
      }
    }
  }
  walk(dir);
  return results;
}

const errors = [];
const files = listFiles(repoDir);

for (const { full, rel } of files) {
  const base = path.basename(full);
  const ext = path.extname(full).toLowerCase();
  const stat = fs.statSync(full);

  if (/^\.env($|\.)/i.test(base) && !/\.example$/i.test(base)) {
    errors.push({ file: rel, rule: "FORBIDDEN_ENV_FILE", detail: `Secret env file: ${base}` });
    continue;
  }
  if (FORBIDDEN_EXTS.has(ext)) {
    errors.push({ file: rel, rule: "FORBIDDEN_BINARY_EXT", detail: `Forbidden ext ${ext}` });
    continue;
  }
  if (stat.size > MAX_FILE_BYTES) {
    errors.push({ file: rel, rule: "FILE_TOO_LARGE", detail: `${(stat.size / 1048576).toFixed(2)} MB > 10 MB` });
    continue;
  }

  const isText = TEXT_EXTS.has(ext) || base === "Dockerfile" || base === "Makefile" || base === "LICENSE";
  if (!isText) continue;

  const content = fs.readFileSync(full, "utf8");
  const isHygieneScript = rel.endsWith("pre-push-hygiene.mjs");
  const isTestOrFixture = rel.includes("tests/") || rel.includes("test_");

  if (!isHygieneScript) {
    for (const pat of SECRET_PATTERNS) {
      const hits = content.match(pat.regex);
      if (hits && hits.length > 0) {
        if (pat.id === "PRIVATE_KEY_BLOCK" && isTestOrFixture && !content.includes("MII")) continue;
        errors.push({ file: rel, rule: pat.id, detail: `Matched ${pat.id} (${hits.length}x)` });
      }
    }
    const pathHits = content.match(LOCAL_PATH_REGEX);
    if (pathHits && pathHits.length > 0) {
      errors.push({ file: rel, rule: "HARDCODED_LOCAL_PATH", detail: `Found ${pathHits.length} local path(s)` });
    }
  }

  if (strictDocs && (ext === ".md" || (ext === ".json" && (rel.startsWith("reports/") || rel.includes("/reports/"))))) {
    const emHits = content.match(/\u2014|\\u2014|&mdash;/gi);
    if (emHits && emHits.length > 0) {
      errors.push({ file: rel, rule: "FORBIDDEN_EM_DASH", detail: `Found ${emHits.length} em-dash(es)` });
    }
  }
}

console.log(`PRE-PUSH-HYGIENE [${path.basename(repoDir)}]: scanned ${files.length} publishable files`);
console.log(`ERRORS: ${errors.length}`);
for (const e of errors) {
  console.error(`  [ERROR] ${e.rule} @ ${e.file} -> ${e.detail}`);
}
if (errors.length > 0) process.exit(1);
console.log("PASS: 0 secrets, 0 bloat, 0 hardcoded local paths.");
