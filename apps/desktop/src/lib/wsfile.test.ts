import { describe, expect, it } from "vitest";

import { wsfileUrl } from "@/lib/wsfile";

describe("wsfileUrl", () => {
  it("addresses a workspace path", () => {
    expect(wsfileUrl("/reports/q3.pdf")).toBe("wsfile://localhost/reports/q3.pdf");
  });

  it("escapes each segment but keeps the separators", () => {
    // Encoding the whole path would escape the slashes and hand the engine one long name.
    expect(wsfileUrl("/a b/q3 (final).pdf")).toBe("wsfile://localhost/a%20b/q3%20(final).pdf");
    expect(wsfileUrl("/한글/사진.png")).toBe(
      "wsfile://localhost/%ED%95%9C%EA%B8%80/%EC%82%AC%EC%A7%84.png",
    );
  });

  it("drops empty segments, so a doubled or trailing slash does not become one", () => {
    expect(wsfileUrl("//a//b/")).toBe("wsfile://localhost/a/b");
  });
});
