const FLASK_BASE_URL = process.env.REACT_APP_FLASK_API || "http://localhost:5000";

async function parseResponse(res) {
  const contentType = res.headers.get("content-type") || "";
  const isJson = contentType.includes("application/json");
  const payload = isJson ? await res.json() : await res.text();

  if (!res.ok) {
    const message =
      (isJson && (payload.error || payload.message)) ||
      (typeof payload === "string" ? payload : "Request failed");
    throw new Error(message);
  }

  return payload;
}

export async function getHealth() {
  const res = await fetch(`${FLASK_BASE_URL}/health`);
  return parseResponse(res);
}

export async function predictPolyp(formData) {
  const res = await fetch(`${FLASK_BASE_URL}/predict`, {
    method: "POST",
    body: formData,
  });
  return parseResponse(res);
}

export async function classifyPolyp(formData) {
  const res = await fetch(`${FLASK_BASE_URL}/classify`, {
    method: "POST",
    body: formData,
  });
  return parseResponse(res);
}

export async function segmentPolyp(formData) {
  const res = await fetch(`${FLASK_BASE_URL}/segment`, {
    method: "POST",
    body: formData,
  });
  return parseResponse(res);
}

export async function predictVideo(formData) {
  const res = await fetch(`${FLASK_BASE_URL}/predict-video`, {
    method: "POST",
    body: formData,
  });
  return parseResponse(res);
}

export async function incrementalUpdate(formData) {
  const res = await fetch(`${FLASK_BASE_URL}/incremental-update`, {
    method: "POST",
    body: formData,
  });
  return parseResponse(res);
}

export async function fineTuneModel() {
  const res = await fetch(`${FLASK_BASE_URL}/fine-tune`, {
    method: "POST",
  });
  return parseResponse(res);
}

export async function setDomain(domainId) {
  const res = await fetch(`${FLASK_BASE_URL}/set-domain`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ domainId }),
  });
  return parseResponse(res);
}

export async function savePatientRecord(patient) {
  const res = await fetch(`${FLASK_BASE_URL}/patients`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(patient),
  });
  return parseResponse(res);
}

export async function listPatientRecords(limit = 50) {
  const res = await fetch(`${FLASK_BASE_URL}/patients?limit=${encodeURIComponent(limit)}`);
  return parseResponse(res);
}

export async function getPatientRecord(patientDbId) {
  const res = await fetch(`${FLASK_BASE_URL}/patients/${encodeURIComponent(patientDbId)}`);
  return parseResponse(res);
}

export async function listPatientEvents(limit = 100) {
  const res = await fetch(`${FLASK_BASE_URL}/patient-events?limit=${encodeURIComponent(limit)}`);
  return parseResponse(res);
}

export async function listPatientEventsById(patientDbId, limit = 100) {
  const res = await fetch(
    `${FLASK_BASE_URL}/patient-events?patientDbId=${encodeURIComponent(patientDbId)}&limit=${encodeURIComponent(limit)}`
  );
  return parseResponse(res);
}

export async function getPatientFullProfile(patientDbId) {
  const res = await fetch(`${FLASK_BASE_URL}/patients/${encodeURIComponent(patientDbId)}/full`);
  return parseResponse(res);
}

export { FLASK_BASE_URL };
