import { useEffect, useMemo, useState } from "react";
import "../styles/Dashboard.css";
import PatientForm, { INITIAL } from "./PatientForm";
import {
  FLASK_BASE_URL,
  classifyPolyp,
  fineTuneModel,
  getPatientFullProfile,
  getPatientRecord,
  incrementalUpdate,
  listPatientEvents,
  listPatientEventsById,
  listPatientRecords,
  predictPolyp,
  savePatientRecord,
  segmentPolyp,
  setDomain,
} from "../api/flaskApi";

const MENU_ITEMS = [
  { id: "overview", label: "Overview" },
  { id: "patient", label: "Patient" },
  { id: "saved-patients", label: "Saved Patients" },
  { id: "predict", label: "Predict" },
  { id: "classify", label: "Classify" },
  { id: "segment", label: "Segment" },
  { id: "history", label: "Patient History" },
];

const PAGE_DESCRIPTIONS = {
  overview: "View current patient status, clinical summary, and recent detection history.",
  patient: "Enter complete patient clinical information used by AI decision endpoints.",
  "saved-patients": "Open stored patient records, full profile details, timeline, and related images.",
  predict: "Run binary screening to detect whether a polyp is present in the uploaded colonoscopy image.",
  classify: "Run multi-class analysis to identify the likely polyp subtype.",
  segment: "Generate a segmentation mask to localize suspicious polyp regions.",
  incremental: "Automatic learning is enabled in backend. Manual update is optional.",
  domain: "Domain adaptation is backend-controlled for simplified workflow.",
  history: "Review recent actions and outcomes for audit and follow-up.",
};

export default function Dashboard() {
  const [activeMenu, setActiveMenu] = useState("overview");

  const [patient, setPatient] = useState(INITIAL);
  const [patientSaved, setPatientSaved] = useState(false);
  const [patientError, setPatientError] = useState("");

  const [predictImage, setPredictImage] = useState(null);
  const [classifyImage, setClassifyImage] = useState(null);
  const [segmentImage, setSegmentImage] = useState(null);
  const [incImage, setIncImage] = useState(null);

  const [predicting, setPredicting] = useState(false);
  const [classifying, setClassifying] = useState(false);
  const [segmenting, setSegmenting] = useState(false);
  const [incSubmitting, setIncSubmitting] = useState(false);
  const [fineTuning, setFineTuning] = useState(false);
  const [settingDomain, setSettingDomain] = useState(false);

  const [predictError, setPredictError] = useState("");
  const [classifyError, setClassifyError] = useState("");
  const [segmentError, setSegmentError] = useState("");
  const [incError, setIncError] = useState("");
  const [fineTuneError, setFineTuneError] = useState("");
  const [domainError, setDomainError] = useState("");

  const [predictResult, setPredictResult] = useState(null);
  const [classifyResult, setClassifyResult] = useState(null);
  const [segmentResult, setSegmentResult] = useState(null);
  const [incResult, setIncResult] = useState(null);
  const [fineTuneResult, setFineTuneResult] = useState(null);
  const [domainResult, setDomainResult] = useState(null);

  const [label, setLabel] = useState("1");
  const [domainId, setDomainId] = useState("0");
  const [dbPatients, setDbPatients] = useState([]);
  const [dbEvents, setDbEvents] = useState([]);
  const [patientsError, setPatientsError] = useState("");
  const [eventsError, setEventsError] = useState("");
  const [selectedPatientId, setSelectedPatientId] = useState("");
  const [selectPatientError, setSelectPatientError] = useState("");
  const [previewImageSrc, setPreviewImageSrc] = useState("");
  const [previewImageTitle, setPreviewImageTitle] = useState("");
  const [viewPatientId, setViewPatientId] = useState("");
  const [viewPatientLoading, setViewPatientLoading] = useState(false);
  const [viewPatientError, setViewPatientError] = useState("");
  const [viewPatientProfile, setViewPatientProfile] = useState(null);

  useEffect(() => {
    loadPatientRecords();
    loadPatientEvents();
  }, []);

  useEffect(() => {
    if (!selectedPatientId && dbPatients.length > 0) {
      onSelectSavedPatient(String(dbPatients[0].id));
    }
  }, [dbPatients, selectedPatientId]);

  const patientSummary = useMemo(() => {
    if (!patient.patientName) return "No patient selected";
    return `${patient.patientName} | Age ${patient.age || "-"} | Risk ${patient.clinicalRisk || "Low"}`;
  }, [patient]);

  const recentDbEvents = useMemo(() => dbEvents.slice(0, 5), [dbEvents]);

  function validatePatientBasics() {
    return patient.patientName && patient.age && patient.gender;
  }

  function addHistory(action, summary) {
    const item = {
      id: Date.now() + Math.floor(Math.random() * 1000),
      createdAt: new Date().toLocaleString(),
      patientName: patient.patientName || "Unknown",
      action,
      summary,
    };
    setDbEvents((prev) => [item, ...prev].slice(0, 100));
  }

  async function handlePatientSave() {
    setPatientError("");
    if (!validatePatientBasics()) {
      setPatientError("Patient Name, Age and Gender are required.");
      return;
    }
    try {
      const data = await savePatientRecord(patient);
      if (data?.patientDbId) {
        setPatient((prev) => ({ ...prev, patientDbId: data.patientDbId }));
        setSelectedPatientId(String(data.patientDbId));
      }
      setPatientSaved(true);
      setTimeout(() => setPatientSaved(false), 2500);
      await loadPatientRecords();
      await loadPatientEvents();
      addHistory("Patient Save", data?.message || "Patient saved to database");
    } catch (err) {
      setPatientError(err.message || "Unable to save patient to database.");
    }
  }

  async function loadPatientRecords() {
    setPatientsError("");
    try {
      const data = await listPatientRecords(50);
      setDbPatients(data?.items || []);
    } catch (err) {
      setPatientsError(err.message || "Unable to load saved patients.");
    }
  }

  async function loadPatientEvents() {
    setEventsError("");
    try {
      const data = await listPatientEvents(100);
      setDbEvents(data?.items || []);
    } catch (err) {
      setEventsError(err.message || "Unable to load patient history.");
    }
  }

  function appendClinicalData(form) {
    Object.entries(patient).forEach(([key, value]) => {
      if (Array.isArray(value)) {
        form.append(key, value.join(", "));
      } else if (value !== undefined && value !== null && value !== "") {
        form.append(key, String(value));
      }
    });
    if (patient.patientDbId) {
      form.append("patientDbId", String(patient.patientDbId));
    }
  }

  function buildImageFormData(file, message) {
    if (!file) {
      throw new Error(message);
    }
    if (!validatePatientBasics()) {
      throw new Error("Please select a saved patient or complete Patient Name, Age and Gender in Patient page first.");
    }

    const form = new FormData();
    form.append("image", file);
    appendClinicalData(form);
    return form;
  }

  async function onPredict() {
    setPredictError("");
    setPredicting(true);
    try {
      const data = await predictPolyp(buildImageFormData(predictImage, "Upload image in Predict page."));
      setPredictResult(data);
      addHistory("Predict", `${data.prediction} (${Math.round((data.confidence || 0) * 100)}%)`);
      await loadPatientEvents();
    } catch (err) {
      setPredictError(err.message || "Prediction failed.");
    } finally {
      setPredicting(false);
    }
  }

  async function onClassify() {
    setClassifyError("");
    setClassifying(true);
    try {
      const data = await classifyPolyp(buildImageFormData(classifyImage, "Upload image in Classify page."));
      setClassifyResult(data);
      addHistory("Classify", `${data.polypType || "Unknown type"} (${data.typeConfidence || 0}%)`);
      await loadPatientEvents();
    } catch (err) {
      setClassifyError(err.message || "Classification failed.");
    } finally {
      setClassifying(false);
    }
  }

  async function onSegment() {
    setSegmentError("");
    setSegmenting(true);
    try {
      if (!segmentImage) throw new Error("Upload image in Segment page.");
      const form = new FormData();
      form.append("image", segmentImage);
      const data = await segmentPolyp(form);
      setSegmentResult(data);
      addHistory("Segment", `Coverage ${data.polypCoverage}% | hasPolyp=${String(data.hasPolyp)}`);
      await loadPatientEvents();
    } catch (err) {
      setSegmentError(err.message || "Segmentation failed.");
    } finally {
      setSegmenting(false);
    }
  }

  async function onIncrementalUpdate() {
    setIncError("");
    setIncSubmitting(true);
    try {
      if (!incImage) throw new Error("Upload image in Incremental page.");
      const form = new FormData();
      form.append("image", incImage);
      form.append("label", label);
      appendClinicalData(form);
      const data = await incrementalUpdate(form);
      setIncResult(data);
      addHistory("Incremental Update", `Buffer ${data.bufferSize} | ready=${String(data.readyToTrain)}`);
      await loadPatientEvents();
    } catch (err) {
      setIncError(err.message || "Incremental update failed.");
    } finally {
      setIncSubmitting(false);
    }
  }

  async function onFineTune() {
    setFineTuneError("");
    setFineTuning(true);
    try {
      const data = await fineTuneModel();
      setFineTuneResult(data);
      addHistory("Fine-tune", data.message || "Fine-tune completed");
      await loadPatientEvents();
    } catch (err) {
      setFineTuneError(err.message || "Fine-tune failed.");
    } finally {
      setFineTuning(false);
    }
  }

  async function onSetDomain() {
    setDomainError("");
    setSettingDomain(true);
    try {
      const data = await setDomain(Number(domainId));
      setDomainResult(data);
      addHistory("Set Domain", data.message || `Domain set to ${domainId}`);
      await loadPatientEvents();
    } catch (err) {
      setDomainError(err.message || "Domain switch failed.");
    } finally {
      setSettingDomain(false);
    }
  }

  async function onSelectSavedPatient(idValue) {
    setSelectPatientError("");
    setSelectedPatientId(idValue);
    if (!idValue) {
      setPatient(INITIAL);
      return;
    }
    try {
      const data = await getPatientRecord(Number(idValue));
      const payload = data?.data || {};
      setPatient({
        ...INITIAL,
        ...payload,
        patientDbId: data?.id ?? Number(idValue),
        patientId: data?.patientId ?? payload.patientId ?? "",
        patientName: data?.patientName ?? payload.patientName ?? "",
        age: String(data?.age ?? payload.age ?? ""),
        gender: data?.gender ?? payload.gender ?? "",
        clinicalRisk: data?.clinicalRisk ?? payload.clinicalRisk ?? "Low",
      });
    } catch (err) {
      setSelectPatientError(err.message || "Unable to load selected patient.");
    }
  }

  function openImagePreview(src, title) {
    setPreviewImageSrc(src);
    setPreviewImageTitle(title || "Image Preview");
  }

  function closeImagePreview() {
    setPreviewImageSrc("");
    setPreviewImageTitle("");
  }

  async function onOpenPatientProfile(idValue) {
    setViewPatientError("");
    if (!idValue) {
      setViewPatientId("");
      setViewPatientProfile(null);
      return;
    }
    setViewPatientId(String(idValue));
    setViewPatientLoading(true);
    try {
      const data = await getPatientFullProfile(Number(idValue));
      setViewPatientProfile(data);
    } catch (err) {
      // Fallback for older backend instances that do not have /patients/{id}/full yet.
      try {
        const [patientData, eventsData] = await Promise.all([
          getPatientRecord(Number(idValue)),
          listPatientEventsById(Number(idValue), 300),
        ]);

        const events = eventsData?.items || [];
        const media = [];
        events.forEach((entry) => {
          const payload = entry?.payload;
          if (payload && typeof payload === "object" && !Array.isArray(payload)) {
            Object.entries(payload).forEach(([key, value]) => {
              if (key.endsWith("Url") && typeof value === "string" && value.trim()) {
                media.push({
                  eventId: entry.id,
                  eventAction: entry.action,
                  eventTime: entry.createdAt,
                  kind: key,
                  url: value,
                  path: value,
                });
              }
            });
          }
        });

        setViewPatientProfile({
          patient: {
            id: patientData?.id,
            patientId: patientData?.patientId,
            patientName: patientData?.patientName,
            age: patientData?.age,
            gender: patientData?.gender,
            clinicalRisk: patientData?.clinicalRisk,
            createdAt: patientData?.createdAt,
            updatedAt: patientData?.updatedAt,
            data: patientData?.data || {},
          },
          events,
          media,
          counts: { events: events.length, media: media.length },
        });
        // Profile loaded successfully via fallback method - no error to report
        setViewPatientError("");
      } catch (fallbackErr) {
        setViewPatientProfile(null);
        setViewPatientError(fallbackErr.message || err.message || "Unable to load patient profile.");
      }
    } finally {
      setViewPatientLoading(false);
    }
  }

  function renderPage() {
    if (activeMenu === "overview") {
      const isPatientEntered = patient.patientName && patient.age;

      return (
        <div className="card">
          <h3 className="card-title">Patient Status</h3>
          <p className="muted-row">{PAGE_DESCRIPTIONS.overview}</p>
          
          {!isPatientEntered ? (
            <div className="empty-state">
              <p className="empty-state-title">No Patient Entered</p>
              <p className="muted-row">Go to <strong>Patient</strong> page to enter clinical information first.</p>
            </div>
          ) : (
            <>
              {recentDbEvents.length > 0 && (
                <div className="patient-status-section" style={{ 
                  borderLeft: "4px solid #8b5cf6", 
                  backgroundColor: "#faf5ff", 
                  padding: "16px",
                  borderRadius: "8px"
                }}>
                  <h4 className="section-title" style={{ color: "#8b5cf6", marginBottom: "12px" }}>Latest Analysis Result</h4>
                  {(() => {
                    const latest = recentDbEvents[0];
                    const isPolyp = latest.summary?.toLowerCase().includes("polyp detected");
                    const isNegative = latest.summary?.toLowerCase().includes("no polyp");
                    const resultColor = isPolyp ? "#dc2626" : isNegative ? "#059669" : "#d97706";
                    const bgColor = isPolyp ? "#fef2f2" : isNegative ? "#f0fdf4" : "#fffbeb";
                    
                    return (
                      <div style={{ 
                        borderLeft: `4px solid ${resultColor}`, 
                        paddingLeft: "16px", 
                        marginBottom: "12px",
                        backgroundColor: bgColor,
                        borderRadius: "6px",
                        padding: "12px 12px 12px 16px"
                      }}>
                        <div style={{ marginBottom: "8px" }}>
                          <span style={{ 
                            fontSize: "11px", 
                            color: "#8b5cf6", 
                            textTransform: "uppercase", 
                            fontWeight: "700",
                            letterSpacing: "0.5px"
                          }}>
                            {latest.action}
                          </span>
                        </div>
                        <div style={{ 
                          fontSize: "18px", 
                          fontWeight: "700", 
                          color: resultColor, 
                          marginBottom: "6px"
                        }}>
                          {latest.summary}
                        </div>
                        <div style={{ 
                          fontSize: "12px", 
                          color: "#8b5cf6", 
                          fontWeight: "500"
                        }}>
                          {latest.createdAt}
                        </div>
                      </div>
                    );
                  })()}
                </div>
              )}

              <div className="patient-status-section">
                <h4 className="section-title">Demographics</h4>
                <div className="overview-grid">
                  <Info label="Name" value={patient.patientName} />
                  <Info label="Age" value={`${patient.age} years`} />
                  <Info label="Gender" value={patient.gender} />
                  <Info label="BMI" value={patient.bmi ? `${Number(patient.bmi).toFixed(1)} kg/m²` : "-"} />
                </div>
              </div>

              <div className="patient-status-section">
                <h4 className="section-title">Clinical Risk Profile</h4>
                <div className="overview-grid">
                  <Info label="Risk Level" value={patient.clinicalRisk || "Low"} />
                  <Info label="Smoking" value={patient.smoking ? "Yes" : "No"} />
                  <Info label="Family History" value={patient.familyHistoryPolyps ? "Positive" : "Negative"} />
                  <Info label="CRP Level" value={patient.crp || "-"} />
                </div>
              </div>

              <div className="patient-status-section">
                <h4 className="section-title">Clinical Vitals</h4>
                <div className="overview-grid">
                  <Info label="Blood Sugar" value={patient.bloodSugar ? `${patient.bloodSugar} mg/dL` : "-"} />
                  <Info label="Symptoms" value={patient.symptoms?.length ? patient.symptoms.join(", ") : "None"} />
                  <Info label="Polyp Location" value={patient.polypLocation || "-"} />
                </div>
              </div>

              <div className="patient-status-section">
                <h4 className="section-title">Database Status</h4>
                <div className="overview-grid">
                  <Info label="Patient DB ID" value={patient.patientDbId || "Not saved yet"} />
                  <Info label="Saved Patients" value={String(dbPatients.length)} />
                  <Info label="Recorded Events" value={String(dbEvents.length)} />
                </div>
              </div>

              <div className="patient-status-section">
                <h4 className="section-title">Recent Detection History</h4>
                {recentDbEvents.length === 0 ? (
                  <p className="muted-row">No recent actions. Start colonoscopy analysis in <strong>Predict</strong> page.</p>
                ) : (
                  <div className="mini-history">
                    {recentDbEvents.map((entry) => (
                      <div key={entry.id} className="history-item">
                        <span className="history-time">{entry.createdAt}</span>
                        <span className="history-action">{entry.action}</span>
                        <span className="history-summary">{entry.summary}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              <div className="action-buttons">
                <button className="btn-secondary" onClick={() => setActiveMenu("patient")}>Edit Patient Info</button>
                <button className="btn-primary" onClick={() => setActiveMenu("predict")}>Start Colonoscopy Detection</button>
              </div>
            </>
          )}
        </div>
      );
    }

    if (activeMenu === "patient") {
      return (
        <div className="card patient-card">
          <h3 className="card-title">Patient</h3>
          <p className="muted-row">{PAGE_DESCRIPTIONS.patient}</p>
          <div className="inline-controls" style={{ marginBottom: 12 }}>
            <label>Select Saved Patient</label>
            <select value={selectedPatientId} onChange={(e) => onSelectSavedPatient(e.target.value)}>
              <option value="">Choose patient...</option>
              {dbPatients.map((row) => (
                <option key={row.id} value={row.id}>
                  {row.patientName} | {row.age} | {row.gender} | ID {row.id}
                </option>
              ))}
            </select>
          </div>
          {selectPatientError && <p className="error-msg">{selectPatientError}</p>}
          <PatientForm
            value={patient}
            onChange={setPatient}
            onSave={handlePatientSave}
            saved={patientSaved}
            error={patientError}
          />

          <div className="patient-status-section">
            <h4 className="section-title">Saved Patients (Database)</h4>
            {patientsError && <p className="error-msg">{patientsError}</p>}
            {dbPatients.length === 0 ? (
              <p className="muted-row">No patient records saved yet. Use Save Patient Info.</p>
            ) : (
              <div className="table-wrap">
                <table className="history-table">
                  <thead>
                    <tr>
                      <th>DB ID</th>
                      <th>Name</th>
                      <th>Age</th>
                      <th>Gender</th>
                      <th>Risk</th>
                      <th>Saved At</th>
                      <th>Use</th>
                    </tr>
                  </thead>
                  <tbody>
                    {dbPatients.map((row) => (
                      <tr key={row.id}>
                        <td>{row.id}</td>
                        <td>{row.patientName}</td>
                        <td>{row.age}</td>
                        <td>{row.gender}</td>
                        <td>{row.clinicalRisk || "Low"}</td>
                        <td>{row.createdAt}</td>
                        <td>
                          <button className="btn-secondary" onClick={() => onSelectSavedPatient(String(row.id))}>
                            Select
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>
      );
    }

    if (activeMenu === "saved-patients") {
      return (
        <div className="card patient-card">
          <h3 className="card-title">Saved Patients</h3>
          <p className="muted-row">{PAGE_DESCRIPTIONS["saved-patients"]}</p>

          <div className="inline-controls" style={{ marginBottom: 12 }}>
            <label>Open Patient Record</label>
            <select value={viewPatientId} onChange={(e) => onOpenPatientProfile(e.target.value)}>
              <option value="">Choose saved patient...</option>
              {dbPatients.map((row) => (
                <option key={row.id} value={row.id}>
                  {row.patientName} | {row.age} | {row.gender} | ID {row.id}
                </option>
              ))}
            </select>
          </div>

          {patientsError && <p className="error-msg">{patientsError}</p>}
          {viewPatientError && <p className="error-msg">{viewPatientError}</p>}
          {viewPatientLoading && <p className="muted-row">Loading patient profile...</p>}

          {dbPatients.length > 0 && (
            <div className="table-wrap" style={{ marginTop: 12 }}>
              <table className="history-table">
                <thead>
                  <tr>
                    <th>DB ID</th>
                    <th>Name</th>
                    <th>Age</th>
                    <th>Gender</th>
                    <th>Risk</th>
                    <th>Saved At</th>
                    <th>Open</th>
                  </tr>
                </thead>
                <tbody>
                  {dbPatients.map((row) => (
                    <tr key={row.id}>
                      <td>{row.id}</td>
                      <td>{row.patientName}</td>
                      <td>{row.age}</td>
                      <td>{row.gender}</td>
                      <td>{row.clinicalRisk || "Low"}</td>
                      <td>{row.createdAt}</td>
                      <td>
                        <button className="btn-secondary" onClick={() => onOpenPatientProfile(String(row.id))}>
                          View
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {viewPatientProfile?.patient && (
            <>
              <div className="patient-status-section" style={{ marginTop: 20 }}>
                <h4 className="section-title">Patient Details</h4>
                <div className="overview-grid">
                  <Info label="DB ID" value={viewPatientProfile.patient.id} />
                  <Info label="Patient ID" value={viewPatientProfile.patient.patientId || "-"} />
                  <Info label="Name" value={viewPatientProfile.patient.patientName} />
                  <Info label="Age" value={String(viewPatientProfile.patient.age || "-")} />
                  <Info label="Gender" value={viewPatientProfile.patient.gender || "-"} />
                  <Info label="Clinical Risk" value={viewPatientProfile.patient.clinicalRisk || "-"} />
                  <Info label="Created At" value={viewPatientProfile.patient.createdAt || "-"} />
                  <Info label="Updated At" value={viewPatientProfile.patient.updatedAt || "-"} />
                </div>
              </div>

              <div className="patient-status-section">
                <h4 className="section-title">Complete Form Data</h4>
                <div className="table-wrap">
                  <table className="history-table">
                    <thead>
                      <tr>
                        <th>Field</th>
                        <th>Value</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(viewPatientProfile.patient.data || {}).map(([key, value]) => (
                        <tr key={key}>
                          <td>{key}</td>
                          <td>{Array.isArray(value) ? value.join(", ") : typeof value === "object" && value !== null ? JSON.stringify(value) : String(value)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              <div className="patient-status-section">
                <h4 className="section-title">Patient Images & Media</h4>
                {!viewPatientProfile.media?.length ? (
                  <p className="muted-row">No media linked yet. Run Predict/Classify/Segment for this patient.</p>
                ) : (
                  <div className="comparison-grid">
                    {viewPatientProfile.media.map((mediaItem, idx) => {
                      const rawUrl = String(mediaItem.url || "");
                      const mediaUrl = rawUrl.startsWith("http://") || rawUrl.startsWith("https://")
                        ? rawUrl
                        : `${FLASK_BASE_URL}${rawUrl}`;
                      const itemPath = String(mediaItem.path || "").toLowerCase();
                      const isImage = /\.(png|jpg|jpeg|webp|gif|bmp)$/i.test(itemPath);
                      return (
                        <div className="info-card" key={`${mediaItem.eventId}-${idx}`}>
                          <span className="info-label">{mediaItem.eventAction} | {mediaItem.kind}</span>
                          {isImage ? (
                            <img
                              src={mediaUrl}
                              alt={mediaItem.kind}
                              className="heatmap-img"
                              onClick={() => openImagePreview(mediaUrl, `${mediaItem.eventAction} - ${mediaItem.kind}`)}
                            />
                          ) : (
                            <a href={mediaUrl} target="_blank" rel="noreferrer" className="btn-secondary" style={{ display: "inline-block", textAlign: "center" }}>
                              Open media
                            </a>
                          )}
                          <span className="muted-row" style={{ margin: 0 }}>{mediaItem.eventTime}</span>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>

              <div className="patient-status-section">
                <h4 className="section-title">Patient Timeline</h4>
                {!viewPatientProfile.events?.length ? (
                  <p className="muted-row">No events recorded for this patient.</p>
                ) : (
                  <div className="table-wrap">
                    <table className="history-table">
                      <thead>
                        <tr>
                          <th>Time</th>
                          <th>Action</th>
                          <th>Summary</th>
                          <th>Payload</th>
                        </tr>
                      </thead>
                      <tbody>
                        {viewPatientProfile.events.map((entry) => (
                          <tr key={entry.id}>
                            <td>{entry.createdAt}</td>
                            <td>{entry.action}</td>
                            <td>{entry.summary}</td>
                            <td>{JSON.stringify(entry.payload || {})}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            </>
          )}
        </div>
      );
    }

    if (activeMenu === "predict") {
      return (
        <div className="card">
          <h3 className="card-title">Colonoscopy Detection</h3>
          <p className="muted-row">{PAGE_DESCRIPTIONS.predict}</p>
          
          <div className="colonoscopy-section">
            <h4 className="section-title">Image Upload</h4>
            <p className="section-desc">Upload a high-quality colonoscopy image (jpeg, png) to screen for polyps.</p>
            <input type="file" accept="image/*" onChange={(e) => setPredictImage(e.target.files?.[0] || null)} />
            {predictImage && <p className="file-selected">✓ Image selected: {predictImage.name}</p>}
          </div>

          <div className="upload-actions">
            <button className="btn-primary" onClick={onPredict} disabled={predicting}>
              {predicting ? "🔄 Processing..." : "Run Polyp Detection"}
            </button>
          </div>

          {predictError && <p className="error-msg">⚠️ {predictError}</p>}

          {predictResult && (
            <div className="result-panel colonoscopy-result">
              <h4 className="section-title">Detection Result</h4>
              {(() => {
                const hasPolyp = String(predictResult.prediction || "").toLowerCase().includes("polyp") &&
                  String(predictResult.prediction || "").toLowerCase() !== "no polyp";
                const recommendation = predictResult.recommendation;
                return (
                  <>
                    <div className="prediction-box" style={{
                      borderLeft: hasPolyp ? "4px solid #ef4444" : "4px solid #10b981"
                    }}>
                      <div className="prediction-content">
                        <span className="prediction-label">Detection:</span>
                        <span className="prediction-value" style={{
                          color: hasPolyp ? "#ef4444" : "#10b981",
                          fontWeight: "bold"
                        }}>
                          {predictResult.prediction}
                        </span>
                      </div>
                      <div className="prediction-content">
                        <span className="prediction-label">Confidence:</span>
                        <span className="prediction-value">{Math.round((predictResult.confidence || 0) * 100)}%</span>
                      </div>
                    </div>
              
                    <div className="overview-grid">
                      <Info label="AI Risk Score" value={predictResult.risk} />
                      <Info label="Final Clinical Risk" value={predictResult.finalRisk} />
                    </div>

                    {predictResult.heatmap && (
                      <div className="preview-block">
                        <h5>Heatmap (AI Attention Region)</h5>
                        <img
                          src={`data:image/jpeg;base64,${predictResult.heatmap}`}
                          alt="heatmap"
                          className="heatmap-img"
                          onClick={() => openImagePreview(`data:image/jpeg;base64,${predictResult.heatmap}`, "Heatmap Preview")}
                        />
                        <p className="muted-row">Click image to view full size.</p>
                      </div>
                    )}

                    {recommendation && (
                      <div className="recommendation-box">
                        <h5>Recommendation</h5>
                        {typeof recommendation === "string" ? (
                          <p>{recommendation}</p>
                        ) : (
                          <>
                            <p><strong>Action:</strong> {recommendation.action || "-"}</p>
                            <p><strong>Urgency:</strong> {recommendation.urgency || "-"}</p>
                            <p><strong>Detail:</strong> {recommendation.detail || "-"}</p>
                            <p><strong>Follow-up:</strong> {recommendation.followup || "-"}</p>
                          </>
                        )}
                      </div>
                    )}
                  </>
                );
              })()}

              <button className="btn-secondary" onClick={() => setActiveMenu("history")}>View in History</button>
            </div>
          )}
        </div>
      );
    }

    if (activeMenu === "classify") {
      return (
        <div className="card">
          <h3 className="card-title">Classify</h3>
          <p className="muted-row">{PAGE_DESCRIPTIONS.classify}</p>
          <input type="file" accept="image/*" onChange={(e) => setClassifyImage(e.target.files?.[0] || null)} />
          <div className="upload-actions">
            <button className="btn-primary" onClick={onClassify} disabled={classifying}>Run Classify</button>
          </div>
          {classifyError && <p className="error-msg">{classifyError}</p>}
          {classifyResult && (
            <div className="result-panel">
              <div className="overview-grid">
                <Info label="Prediction" value={classifyResult.prediction} />
                <Info label="Polyp Type" value={classifyResult.polypType} />
                <Info label="Type Confidence" value={`${classifyResult.typeConfidence || 0}%`} />
              </div>
            </div>
          )}
        </div>
      );
    }

    if (activeMenu === "segment") {
      return (
        <div className="card">
          <h3 className="card-title">Segment</h3>
          <p className="muted-row">{PAGE_DESCRIPTIONS.segment}</p>
          <input type="file" accept="image/*" onChange={(e) => setSegmentImage(e.target.files?.[0] || null)} />
          <div className="upload-actions">
            <button className="btn-primary" onClick={onSegment} disabled={segmenting}>Run Segment</button>
          </div>
          {segmentError && <p className="error-msg">{segmentError}</p>}
          {segmentResult && (
            <div className="result-panel">
              <div className="overview-grid">
                <Info label="Coverage" value={`${segmentResult.polypCoverage}%`} />
                <Info label="Has Polyp" value={String(segmentResult.hasPolyp)} />
              </div>
              {segmentResult.segmentationMask && (
                <div className="preview-block">
                  <img
                    src={`data:image/jpeg;base64,${segmentResult.segmentationMask}`}
                    alt="segmentation"
                    className="heatmap-img"
                    onClick={() => openImagePreview(`data:image/jpeg;base64,${segmentResult.segmentationMask}`, "Segmentation Preview")}
                  />
                  <p className="muted-row">Click image to view full size.</p>
                </div>
              )}
            </div>
          )}
        </div>
      );
    }

    if (activeMenu === "incremental") {
      return (
        <div className="card">
          <h3 className="card-title">Incremental Learning</h3>
          <p className="muted-row">{PAGE_DESCRIPTIONS.incremental}</p>
          <input type="file" accept="image/*" onChange={(e) => setIncImage(e.target.files?.[0] || null)} />
          <div className="inline-controls">
            <label>Confirmed Label</label>
            <select value={label} onChange={(e) => setLabel(e.target.value)}>
              <option value="1">1 - Polyp</option>
              <option value="0">0 - No Polyp</option>
            </select>
          </div>
          <div className="upload-actions">
            <button className="btn-primary" onClick={onIncrementalUpdate} disabled={incSubmitting}>Add To Replay Buffer</button>
            <button className="btn-secondary" onClick={onFineTune} disabled={fineTuning}>Run Fine-tune</button>
          </div>
          {incError && <p className="error-msg">{incError}</p>}
          {fineTuneError && <p className="error-msg">{fineTuneError}</p>}
          {incResult && <p className="success-msg">{incResult.message} | Buffer: {incResult.bufferSize}</p>}
          {fineTuneResult && <p className="success-msg">{fineTuneResult.message}</p>}
        </div>
      );
    }

    if (activeMenu === "domain") {
      return (
        <div className="card">
          <h3 className="card-title">Domain Adapter</h3>
          <p className="muted-row">{PAGE_DESCRIPTIONS.domain}</p>
          <div className="inline-controls">
            <label>Domain ID</label>
            <input type="number" min="0" max="15" value={domainId} onChange={(e) => setDomainId(e.target.value)} />
          </div>
          <button className="btn-primary" onClick={onSetDomain} disabled={settingDomain}>Set Domain</button>
          {domainError && <p className="error-msg">{domainError}</p>}
          {domainResult && <p className="success-msg">{domainResult.message}</p>}
        </div>
      );
    }

    return (
      <div className="card">
        <h3 className="card-title">Patient History</h3>
        <p className="muted-row">{PAGE_DESCRIPTIONS.history}</p>
        {eventsError && <p className="error-msg">{eventsError}</p>}
        {dbEvents.length === 0 ? (
          <p className="muted-row">No history yet. Save a patient and run any feature to generate records.</p>
        ) : (
          <div className="table-wrap">
            <table className="history-table">
              <thead>
                <tr>
                  <th>Time</th>
                  <th>Patient</th>
                  <th>Action</th>
                  <th>Summary</th>
                </tr>
              </thead>
              <tbody>
                {dbEvents.map((entry) => (
                  <tr key={entry.id}>
                    <td>{entry.createdAt}</td>
                    <td>{entry.patientName || "-"}</td>
                    <td>{entry.action}</td>
                    <td>{entry.summary}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    );
  }

  const currentPageLabel = MENU_ITEMS.find((item) => item.id === activeMenu)?.label || "Dashboard";

  return (
    <div className="dash-wrapper">
      <aside className="sidebar">
        <div className="sidebar-logo">
          <h2>Polyp Assist</h2>
        </div>
        <nav className="sidebar-nav">
          {MENU_ITEMS.map((item) => (
            <button
              key={item.id}
              className={`nav-btn ${activeMenu === item.id ? "active" : ""}`}
              onClick={() => setActiveMenu(item.id)}
            >
              {item.label}
            </button>
          ))}
        </nav>
      </aside>

      <main className="dash-main">
        <header className="dash-header">
          <h1 className="dash-title">{currentPageLabel}</h1>
          <div className="doctor-badge">{patientSummary} | Saved Patients: {dbPatients.length}</div>
        </header>
        <div className="inline-controls" style={{ marginBottom: 12 }}>
          <label>Active Patient</label>
          <select value={selectedPatientId} onChange={(e) => onSelectSavedPatient(e.target.value)}>
            <option value="">Choose saved patient for upload...</option>
            {dbPatients.map((row) => (
              <option key={row.id} value={row.id}>
                {row.patientName} | {row.age} | {row.gender} | ID {row.id}
              </option>
            ))}
          </select>
        </div>
        {selectPatientError && <p className="error-msg">{selectPatientError}</p>}
        <div className="dash-content">{renderPage()}</div>

        {previewImageSrc && (
          <div className="image-modal-backdrop" onClick={closeImagePreview}>
            <div className="image-modal-content" onClick={(e) => e.stopPropagation()}>
              <div className="image-modal-header">
                <h4>{previewImageTitle}</h4>
                <button className="btn-secondary" onClick={closeImagePreview}>Close</button>
              </div>
              <div className="image-modal-body">
                <img src={previewImageSrc} alt={previewImageTitle} className="image-modal-img" />
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

function Info({ label, value }) {
  return (
    <div className="info-card">
      <span className="info-label">{label}</span>
      <span className="info-value">{value || "-"}</span>
    </div>
  );
}
