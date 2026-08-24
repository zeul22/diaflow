import { describe, expect, it } from "vitest";

import { isAnalysisResponse, isLanguagePrediction, languageName } from "./analyze.js";

const baseResult = {
  contact_id: "123e4567-e89b-12d3-a456-426614174000",
  gender: { prediction: "male", confidence: 0.9 },
  age_bracket: { prediction: "31-45", confidence: 0.5 },
  processing_ms: 200,
  audio_quality: "good",
};

describe("languageName", () => {
  it("names known tags and falls back to the tag itself", () => {
    expect(languageName("en")).toBe("English");
    expect(languageName("th")).toBe("Thai");
    // A 107-language model can return tags this UI does not enumerate.
    expect(languageName("ceb")).toBe("CEB");
    expect(languageName("unknown")).toBe("Not determined");
    expect(languageName("")).toBe("—");
    expect(languageName(undefined)).toBe("—");
  });
});

describe("isLanguagePrediction", () => {
  it("accepts a tag or unknown with a bounded confidence", () => {
    expect(isLanguagePrediction({ prediction: "en", confidence: 0.47 })).toBe(true);
    expect(isLanguagePrediction({ prediction: "unknown", confidence: 0 })).toBe(true);
  });

  it("rejects malformed values rather than rendering them", () => {
    expect(isLanguagePrediction({ prediction: "English", confidence: 0.9 })).toBe(false);
    expect(isLanguagePrediction({ prediction: "en", confidence: 1.4 })).toBe(false);
    expect(isLanguagePrediction({ prediction: "en" })).toBe(false);
    expect(isLanguagePrediction(null)).toBe(false);
  });
});

describe("isAnalysisResponse language handling", () => {
  it("treats language as optional so a deployment can run without it", () => {
    expect(isAnalysisResponse(baseResult)).toBe(true);
    expect(isAnalysisResponse({ ...baseResult, language: null })).toBe(true);
    expect(
      isAnalysisResponse({
        ...baseResult,
        language: { prediction: "hi", confidence: 0.88 },
      }),
    ).toBe(true);
  });

  it("rejects a response whose language field is malformed", () => {
    expect(
      isAnalysisResponse({ ...baseResult, language: { prediction: "Hindi" } }),
    ).toBe(false);
  });
});
