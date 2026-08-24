import { describe, expect, it, vi } from "vitest";

import {
  deleteStoredAnalysis,
  getPersistenceCapabilities,
  getStoredAnalysis,
  listStoredAnalyses,
  persistenceModeIsAvailable,
  persistenceHeaders,
  persistenceStartFields,
} from "./persistence.js";

describe("persistence API", () => {
  it("defaults to no persistence and only includes a consent reference for stored audio", () => {
    expect(persistenceHeaders()).toEqual({ "X-Persistence-Mode": "none" });
    expect(persistenceHeaders({ mode: "result", consentReference: "ignored" })).toEqual({
      "X-Persistence-Mode": "result",
    });
    expect(
      persistenceStartFields({
        mode: "result_and_audio",
        consentReference: "  consent-ticket-8  ",
      }),
    ).toEqual({
      persistence_mode: "result_and_audio",
      consent_reference: "consent-ticket-8",
    });
    expect(persistenceModeIsAvailable("result", "result_and_audio")).toBe(true);
    expect(persistenceModeIsAvailable("result_and_audio", "result")).toBe(false);
  });

  it("loads and validates the deployment persistence maximum", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: vi.fn().mockResolvedValue({
        enabled: true,
        maximum_mode: "result",
        default_mode: "result",
        audio_retention_hours: 24,
      }),
    });
    vi.stubGlobal("fetch", fetchMock);

    await expect(getPersistenceCapabilities()).resolves.toMatchObject({
      enabled: true,
      maximum_mode: "result",
      default_mode: "none",
    });
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/v1/persistence/capabilities",
      expect.objectContaining({ cache: "no-store" }),
    );
  });

  it("normalizes history envelopes and deletes an encoded analysis ID", async () => {
    const analyses = [{ analysis_id: "analysis/8", status: "completed" }];
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: vi.fn().mockResolvedValue({ items: analyses }),
      })
      .mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: vi.fn().mockResolvedValue({ ...analyses[0], segments: [] }),
      })
      .mockResolvedValueOnce({ ok: true, status: 204 });
    vi.stubGlobal("fetch", fetchMock);

    await expect(listStoredAnalyses()).resolves.toEqual(analyses);
    await expect(getStoredAnalysis("analysis/8")).resolves.toEqual({
      ...analyses[0],
      segments: [],
    });
    await deleteStoredAnalysis("analysis/8");

    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      "/api/v1/analyses",
      expect.objectContaining({ cache: "no-store" }),
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      "/api/v1/analyses/analysis%2F8",
      expect.objectContaining({ cache: "no-store" }),
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      3,
      "/api/v1/analyses/analysis%2F8",
      expect.objectContaining({ method: "DELETE" }),
    );
  });
});
