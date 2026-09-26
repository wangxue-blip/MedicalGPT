import { readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const input = resolve(here, "MedicalGPT_A40_4GPU_复现与面试指南_详细扩展版.html");
const output = resolve(here, "MedicalGPT_A40_4GPU_复现与面试指南_详细扩展版_pdf_source.html");

const html = readFileSync(input, "utf8")
  .replace("Data Analytics report", "技术报告")
  .replace(
    "</head>",
    `<style media="print">
      @page { size: A4; margin: 13mm 12mm 15mm; }
      .portable-page-header { break-after: avoid; }
      .portable-table-card, .portable-chart-card { break-inside: avoid-page; }
      pre { white-space: pre-wrap !important; overflow: visible !important; max-width: 100% !important; }
      pre code, code { white-space: pre-wrap !important; overflow-wrap: anywhere; word-break: break-word; }
    </style></head>`,
  );

writeFileSync(output, html, "utf8");
console.log(output);
